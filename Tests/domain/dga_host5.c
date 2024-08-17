#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  // 确保包含 stdint.h
#include <dpu_management.h>


#ifndef DPU_BINARY
#define DPU_BINARY "./dgaa"  
#endif

#define MAX_DOMAIN_LEN 64  // Host 端的定义要和 DPU 端一致
#define VOCAB_SIZE (MAX_DOMAIN_LEN * sizeof(int32_t))  // 计算 vocab_size 的大小
#define MODEL_SIZE 1814336
#define NR_TASKLETS 11

typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t label;
    int32_t padding[6]; // 确保结构体大小为8的倍数
} Example;

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

void transfer_vocabulary(struct dpu_set_t set, int *vocab) {
    struct dpu_set_t dpu;
    size_t vocab_size = MAX_DOMAIN_LEN * sizeof(int32_t);  // 根据新的 MAX_DOMAIN_LEN 计算大小

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
void transfer_domains(struct dpu_set_t set, char **domains, size_t total_domain_chars_size) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        while (offset < total_domain_chars_size) {
            size_t chunk_size = (total_domain_chars_size - offset > 2048) ? 2048 : (total_domain_chars_size - offset);
            if (chunk_size % 8 != 0) {
                chunk_size -= (chunk_size % 8); // 调整chunk_size对齐到8字节
            }
            check_alignment(offset, chunk_size);
            // 将 "domains" 修改为 "mram_domains"
            DPU_ASSERT(dpu_copy_to(dpu, "mram_domains", offset, (void*)((uintptr_t)domains + offset), chunk_size));
            offset += chunk_size;
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
    size_t vocab_size = sizeof(vocab);

    // 确保 vocab_size 是 8 字节对齐的
    if (vocab_size % 8 != 0) {
        vocab_size += (8 - (vocab_size % 8));
    }

    printf("Transferring vocabulary to DPU MRAM...\n");
    //transfer_vocabulary(set, vocab, vocab_size);

    // 域名列表并确保对齐
    char *domains_need_predict[] = {
        "baidu.com", "01ol.ee4kdushuba.com", "nhy655.3322.org",
        "tmall.com", "office.com", "taobao.com", "dropbox.com"
    };
    
    // 计算实际字符数据的总大小
    size_t total_domain_chars_size = 0;
    for (size_t i = 0; i < sizeof(domains_need_predict)/sizeof(domains_need_predict[0]); i++) {
        total_domain_chars_size += strlen(domains_need_predict[i]) + 1; // +1 是为了包含空终止符
    }

    // 确保总大小是8字节对齐的
    if (total_domain_chars_size % 8 != 0) {
        total_domain_chars_size += (8 - (total_domain_chars_size % 8));
    }

    printf("Total size of domain characters: %zu\n", total_domain_chars_size);
    printf("Transferring domain names to DPU MRAM...\n");
    transfer_domains(set, domains_need_predict, total_domain_chars_size);

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
        DPU_ASSERT(dpu_copy_from(dpu, "results", 0, results, sizeof(results)));
    }
    printf("Results read successfully.\n");

    // 打印结果
    printf("Results:\n");
    for (int i = 0; i < NR_TASKLETS; i++) {
        int label = results[i].label;
        printf("Domain: %s, Label: %d\n", results[i].domain_name, label);
        if (label == 1) {
            printf("%s 检测结果为 : 恶意地址\n", results[i].domain_name);
        } else {
            printf("%s 检测结果为 : 非恶意地址\n", results[i].domain_name);
        }
    }

    printf("Freeing DPU set...\n");
    DPU_ASSERT(dpu_free(set));
    printf("DPU set freed successfully.\n");

    return 0;
}
