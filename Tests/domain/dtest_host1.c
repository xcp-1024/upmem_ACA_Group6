#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  // 确保包含 stdint.h
#include <dpu_management.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./dtest"  
#endif

#define MAX_DOMAIN_LEN 64  // Host 端的定义要和 DPU 端一致
#define VOCAB_SIZE (MAX_DOMAIN_LEN * sizeof(int32_t))  // 计算 vocab_size 的大小
#define MODEL_SIZE 1814336
#define NR_TASKLETS 11
#define DOMAIN_MEM_SIZE 1024  // 定义 DPU 端 MRAM 中域名字符存储的大小
#define RESULTS_SIZE (NR_TASKLETS * sizeof(Example))  // 结果数组的大小
#define LSTM_HIDDEN_SIZE 50  // 根据你的 LSTM 隐藏层的大小来设置


typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;                 // 偏移量 0, 对齐
    int32_t padding1;                   // 填充到 8 字节对齐
    char domain_name[MAX_DOMAIN_LEN];   // 偏移量 8, 对齐
    int32_t padding2;                   // 填充到 8 字节对齐
    int32_t char_ids[MAX_DOMAIN_LEN];   // 偏移量 176, 对齐
    int32_t label;                      // 偏移量 832, 对齐
    int32_t padding3[4];                // 确保结构体大小为8的倍数，避免对齐问题
} Example;









typedef struct __attribute__((aligned(8))) {
    int32_t char_ids[MAX_DOMAIN_LEN];
    float lstm_output[LSTM_HIDDEN_SIZE];
    float dense_output[2];
} DebugInfo;

// 定义 get_char_id 函数
int get_char_id(char c) {
    return (int)c;  // 示例：使用字符的 ASCII 值作为 ID
}


// 打印从 MRAM 中读取的域名字符数据
void read_and_print_domains_from_mram(struct dpu_set_t set) {
    struct dpu_set_t dpu;
    char mram_domain[MAX_DOMAIN_LEN];

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "mram_domains", 0, mram_domain, MAX_DOMAIN_LEN));
        printf("Domain from MRAM: %s\n", mram_domain);
    }
}

// 定义函数来打印 DebugInfo 结构的内容
void print_debug_info(DebugInfo *debug_info) {
    printf("Character IDs: ");
    for (int i = 0; i < MAX_DOMAIN_LEN; i++) {
        printf("%d ", debug_info->char_ids[i]);
    }
    printf("\nLSTM Output: ");
    for (int i = 0; i < LSTM_HIDDEN_SIZE; i++) {
        printf("%f ", debug_info->lstm_output[i]);
    }
    printf("\nDense Output: %f %f\n", debug_info->dense_output[0], debug_info->dense_output[1]);
}

// 检查DPU分配是否成功
void check_allocation(struct dpu_set_t *set) {
    uint32_t nr_of_dpus;
    DPU_ASSERT(dpu_get_nr_dpus(*set, &nr_of_dpus));
    if (nr_of_dpus == 0) {
        printf("DPU allocation failed.\n");
        exit(EXIT_FAILURE);
    }
}

// 检查偏移量和大小是否是8字节对齐的
void check_alignment(size_t offset, size_t size) {
    if (offset % 8 != 0 || size % 8 != 0) {
        printf("ERROR: Offset or size is not 8-byte aligned! Offset: %zu, Size: %zu\n", offset, size);
        exit(EXIT_FAILURE); // 如果不对齐，退出程序
    }
}

// 分块传输vocab数组
void transfer_vocabulary(struct dpu_set_t set, int *vocab) {
    struct dpu_set_t dpu;
    size_t vocab_size = VOCAB_SIZE;  // 使用预定义的 VOCAB_SIZE

    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        while (offset < vocab_size) {
            size_t chunk_size = (vocab_size - offset > 2048) ? 2048 : (vocab_size - offset);
            if (chunk_size % 8 != 0) {
                chunk_size -= (chunk_size % 8); // 调整chunk_size对齐到8字节
            }
            check_alignment(offset, chunk_size);
            DPU_ASSERT(dpu_copy_to(dpu, "mram_vocabulary", offset, (void*)((uintptr_t)vocab + offset), chunk_size));
            offset += chunk_size;
        }
    }
    printf("Vocabulary transferred successfully.\n");
}

// 分块传输域名字符数据
void transfer_domains(struct dpu_set_t set, char **domains) {
    struct dpu_set_t dpu;
    size_t total_domain_chars_size = 0;

    // 计算总的域名字符大小并处理对齐
    for (size_t i = 0; i < sizeof(domains) / sizeof(domains[0]); i++) {
        size_t domain_len = strlen(domains[i]) + 1; // 包括终止符
        size_t aligned_len = domain_len;

        // 确保对齐到 8 字节
        if (aligned_len % 8 != 0) {
            aligned_len += (8 - (aligned_len % 8));
        }

        total_domain_chars_size += aligned_len;
    }

    // 确保总大小是8字节对齐的
    if (total_domain_chars_size % 8 != 0) {
        total_domain_chars_size += (8 - (total_domain_chars_size % 8));
    }

    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        for (size_t i = 0; i < sizeof(domains) / sizeof(domains[0]); i++) {
            size_t domain_len = strlen(domains[i]) + 1; // 包括终止符
            size_t aligned_len = domain_len;

            // 确保每个域名对齐到 8 字节
            if (aligned_len % 8 != 0) {
                aligned_len += (8 - (aligned_len % 8));
            }

            char aligned_domain[MAX_DOMAIN_LEN] = {0};  // 初始化填充的对齐域名
            strncpy(aligned_domain, domains[i], domain_len);

            check_alignment(offset, aligned_len);
            DPU_ASSERT(dpu_copy_to(dpu, "mram_domains", offset, aligned_domain, aligned_len));
            offset += aligned_len;
        }
    }
    printf("Domain names transferred successfully.\n");
}


// 分块传输模型权重
void load_model_weights_to_mram(struct dpu_set_t set, void *model_data) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        while (offset < MODEL_SIZE) {
            size_t chunk_size = (MODEL_SIZE - offset > 2048) ? 2048 : (MODEL_SIZE - offset);
            if (chunk_size % 8 != 0) {
                chunk_size -= (chunk_size % 8); // 确保 chunk_size 是 8 的倍数
            }
            check_alignment(offset, chunk_size);
            printf("Copying model weights to DPU at offset %zu (chunk size %zu)\n", offset, chunk_size);
            DPU_ASSERT(dpu_copy_to(dpu, "mram_model_weights", offset, (void*)((uintptr_t)model_data + offset), chunk_size));
            offset += chunk_size;
        }
    }
    printf("Model weights loaded into MRAM.\n");
}

void print_inference_results(Example *results, int num_results) {
    printf("Inference Results:\n");

    int final_result = 0;  // 用于保存所有 Tasklet 的累加结果

    for (int i = 0; i < num_results; i++) {
        int label = results[i].label;
        printf("Domain: %s, Label: %d\n", results[i].domain_name, label);

        if (label != 0) {
            final_result = 1;  // 只要有一个 label 是 1，就将 final_result 置为 1
        }
    }

    // 根据累加结果进行最终判断
    if (final_result == 1) {
        printf("检测结果为: 恶意地址存在\n");
    } else {
        printf("检测结果为: 没有发现恶意地址\n");
    }
}


int main() {
        struct dpu_set_t set;
    struct dpu_set_t dpu;

    printf("Allocating DPU set...\n");
    DPU_ASSERT(dpu_alloc(1, "backend=simulator", &set));
    check_allocation(&set);
    printf("DPU set allocated successfully.\n");

    printf("Loading DPU binary: %s\n", DPU_BINARY);
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    // 初始化 vocab 并确保对齐
    int vocab[MAX_DOMAIN_LEN] = {0};

    printf("Transferring vocabulary to DPU MRAM...\n");
    transfer_vocabulary(set, vocab);

    // 域名列表并确保对齐
    char *domains_need_predict[] = {
        "baisad55du.comdddddddda"
    };
    printf("Transferring domain names to DPU MRAM...\n");
    transfer_domains(set, domains_need_predict);

        // 在传输数据到 DPU MRAM 之前，打印 Example 结构体内容
    Example example;
    memset(&example, 0, sizeof(Example)); // 初始化结构体

    // 假设域名为 "01ol.ee4kdushuba.com"，并已转换为对应的 char_ids
    example.domain_len = strlen(domains_need_predict[0]);
    strncpy(example.domain_name, domains_need_predict[0], MAX_DOMAIN_LEN);

    // 填充 char_ids 数组 (假设有一个映射函数 get_char_id)
    for (int i = 0; i < example.domain_len; i++) {
        example.char_ids[i] = get_char_id(domains_need_predict[0][i]);
    }

    // 打印 Example 结构体的内容
    printf("Before transferring to MRAM:\n");
    printf("Domain Length: %d\n", example.domain_len);
    printf("Domain Name: %s\n", example.domain_name);
    printf("Char IDs: ");
    for (int i = 0; i < MAX_DOMAIN_LEN; i++) {
        printf("%d ", example.char_ids[i]);
    }
    printf("\n");

    // 现在将填充的 Example 结构体数据传输到 DPU MRAM
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "mram_examples", 0, &example, sizeof(Example)));
    }


    // 在这里添加验证代码
    DPU_FOREACH(set, dpu) {
    // 在 MRAM 写入之后，立即从 MRAM 中读取数据进行验证
    Example verify_example;
    DPU_ASSERT(dpu_copy_from(dpu, "mram_examples", 0, &verify_example, sizeof(Example)));

    // 打印读取到的结构体数据
    printf("Verification - Domain Length: %d\n", verify_example.domain_len);
    printf("Verification - Domain Name: %s\n", verify_example.domain_name);
    printf("Verification - Char IDs: ");
    for (int i = 0; i < MAX_DOMAIN_LEN; i++) {
        printf("%d ", verify_example.char_ids[i]);
    }
    printf("\n");
}

   // **在这里添加域名字符数据读取和验证**
    printf("Reading and printing domain names from DPU MRAM...\n");
    read_and_print_domains_from_mram(set);
   

    // 加载模型权重到 MRAM
    printf("Loading model weights from file...\n");
    FILE *model_file = fopen("/mnt/g/upmem/upmem/upmem_ACA-main/model_saved/pytorch_model_1.bin", "rb");
    if (!model_file) {
        perror("Failed to open model weights file");
        exit(EXIT_FAILURE);
    }

    // 使用 malloc 和手动对齐替代 posix_memalign
    void *model_data_unaligned = malloc(MODEL_SIZE + 8); // 多分配8字节用于对齐
    if (!model_data_unaligned) {
        perror("Failed to allocate memory for model data");
        fclose(model_file);
        exit(EXIT_FAILURE);
    }

    // 手动对齐到8字节边界
    void *model_data = (void*)(((uintptr_t)model_data_unaligned + 7) & ~((uintptr_t)7));

    size_t read_size = fread(model_data, 1, MODEL_SIZE, model_file);
    if (read_size != MODEL_SIZE) {
        perror("Failed to read the complete model data");
        free(model_data_unaligned);  // 释放未对齐的内存
        fclose(model_file);
        exit(EXIT_FAILURE);
    }
    fclose(model_file);
    printf("Model weights loaded from file successfully.\n");

    printf("Transferring model weights to DPU MRAM...\n");
    load_model_weights_to_mram(set, model_data);
    free(model_data_unaligned);  // 释放未对齐的内存

    // 启动 DPU 程序
    printf("Launching DPU program...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
    }
    printf("DPU program launched successfully.\n");

    // 从 MRAM 读取结果
    Example results[NR_TASKLETS];
    printf("Reading results from DPU MRAM...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "mram_examples", 0, results, sizeof(results)));
    }
    printf("Results read successfully.\n");
    
    /*// 从 MRAM 读取调试信息
    DebugInfo debug_results[NR_TASKLETS];
    printf("Reading debug information from DPU MRAM...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "mram_debug_info", 0, debug_results, sizeof(debug_results)));
    }
    printf("Debug information read successfully.\n");

    // 打印详细调试信息
    for (int i = 0; i < NR_TASKLETS; i++) {
        printf("Tasklet %d Debug Info:\n", i);
        print_debug_info(&debug_results[i]);
    }*/
   


    // 打印详细结果
    print_inference_results(results, NR_TASKLETS);

    

    printf("Freeing DPU set...\n");
    DPU_ASSERT(dpu_free(set));
    printf("DPU set freed successfully.\n");

    return 0;
}