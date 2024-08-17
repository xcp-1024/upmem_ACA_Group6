// 替换 wram_alloc 和 wram_free 的部分
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <barrier.h>
#include <mutex.h>
#include <sem.h>
#include <handshake.h>
#include <mram.h>
#include <alloc.h>   // 替代 heap.h，使用 alloc.h 进行内存管理

#define NR_TASKLETS 11
#define MAX_DOMAIN_LEN 130   // 更新为配置文件中的最大长度
#define EMBEDDING_DIM 128
#define LSTM_HIDDEN_SIZE 100
#define LSTM_LAYERS 2        // 根据配置文件中的层数
#define BIDIRECTIONAL 1      // 双向LSTM的标志
#define DENSE_SIZE 100       // 全连接层大小
#define DPU_XFER_TO_MRAM 0x1  // 这个定义通常由dpu.h提供，但可以手动定义以防未定义
#define MODEL_SIZE 1814336 // compute by python

// 定义同步变量
BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(my_mutex);
SEMAPHORE_INIT(my_semaphore, 1);  // 使用 sem.h 中的 SEMAPHORE_INIT

// 定义模型参数和DPU函数
// Example 结构体的大小可能不是8的倍数，需要进行调整
typedef struct {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t label;
    int32_t padding[2]; // 确保结构体大小为8的倍数
} Example;

struct dpu_set_t dpu;

__mram_noinit int32_t mram_vocabulary[MAX_DOMAIN_LEN];
__mram_noinit float mram_model_weights[1814336 / sizeof(float)]; // 使用float明确数据类型
__host int32_t nr_domains;
__host char domains[MAX_DOMAIN_LEN * NR_TASKLETS];

// 模型加载函数
// 替换 load_model_weights_to_mram 函数中的 mram_write
void load_model_weights_to_mram(struct dpu_set_t dpu_set) {
    // 打开模型权重文件
    FILE *model_file = fopen("/mnt/g/upmem/upmem/upmem_ACA-main/model_saved/pytorch_model_1.bin", "rb");
    if (!model_file) {
        perror("Failed to open model weights file");
        exit(EXIT_FAILURE);
    }

    // 为模型权重分配内存
    void *model_data = malloc(MODEL_SIZE);
    if (!model_data) {
        perror("Failed to allocate memory for model data");
        fclose(model_file);
        exit(EXIT_FAILURE);
    }

    // 读取模型权重数据到主机内存
    size_t read_size = fread(model_data, 1, MODEL_SIZE, model_file);
    if (read_size != MODEL_SIZE) {
        perror("Failed to read the complete model data");
        free(model_data);
        fclose(model_file);
        exit(EXIT_FAILURE);
    }
    fclose(model_file);

    // 将模型权重数据写入到每个DPU的MRAM，分块写入
    DPU_FOREACH(dpu_set, dpu) {
        __mram_ptr void *mram_dest = (__mram_ptr void *)0x0; // MRAM的目标地址（从0开始）

        size_t offset = 0;
        while (offset < MODEL_SIZE) {
            size_t chunk_size = (MODEL_SIZE - offset > 2048) ? 2048 : (MODEL_SIZE - offset);
            mram_write((const void *)((uintptr_t)model_data + offset), (__mram_ptr void *)((uintptr_t)mram_dest + offset), chunk_size);
            offset += chunk_size;
        }
    }

    free(model_data);
}

// Tanh 的近似实现
float tanh_approx(float x) {
    if (x < -3.0f) return -1.0f;
    else if (x > 3.0f) return 1.0f;
    else return x * (27.0f + x * x) / (27.0f + 9.0f * x * x);
}

// Exp 的近似实现
float exp_approx(float x) {
    float result = 1.0f;
    float term = 1.0f;
    int n;

    for (n = 1; n < 10; n++) {
        term *= x / n;
        result += term;
    }
    return result;
}

void run_inference(Example *example) {
    int32_t char_ids[MAX_DOMAIN_LEN];
    __mram_ptr void *char_ids_mram = (__mram_ptr void *)&example->char_ids;

    size_t offset = 0;
    while (offset < sizeof(char_ids)) {
        size_t chunk_size = (sizeof(char_ids) - offset > 2048) ? 2048 : (sizeof(char_ids) - offset);
        mram_read((__mram_ptr void*)((uintptr_t)char_ids_mram + offset), (void*)((uintptr_t)char_ids + offset), chunk_size);
        offset += chunk_size;
    }

    // 动态分配嵌入层内存
    float (*embeddings)[EMBEDDING_DIM] = mem_alloc(MAX_DOMAIN_LEN * EMBEDDING_DIM * sizeof(float));
    if (embeddings == NULL) {
        printf("Failed to allocate memory for embeddings\n");
        return;
    }

    // 嵌入层计算
    for (int32_t i = 0; i < example->domain_len; i++) {
        int32_t char_id = char_ids[i];
        for (int32_t j = 0; j < EMBEDDING_DIM; j++) {
            embeddings[i][j] = ((float*)mram_model_weights)[char_id * EMBEDDING_DIM + j];
        }
    }

    // LSTM编码器计算
    float lstm_output[LSTM_HIDDEN_SIZE] = {0};
    float lstm_backward_output[LSTM_HIDDEN_SIZE] = {0};

    for (int32_t layer = 0; layer < LSTM_LAYERS; layer++) {
        printf("Tasklet: LSTM Layer %d processing...\n", layer);
        int32_t reverse_index;
        for (int32_t i = 0; i < example->domain_len; i++) {
            reverse_index = example->domain_len - i - 1;
            for (int32_t j = 0; j < LSTM_HIDDEN_SIZE; j++) {
                float forward_val = embeddings[i][j];
                float backward_val = embeddings[reverse_index][j];

                lstm_output[j] += forward_val;
                lstm_backward_output[j] += backward_val;
            }
        }

        if (BIDIRECTIONAL) {
            for (int32_t j = 0; j < LSTM_HIDDEN_SIZE; j++) {
                lstm_output[j] += lstm_backward_output[j];
            }
        }
    }

    // 全连接层和Softmax处理
    float dense_output[2] = {0}; // 确保此变量在此处定义

    printf("Tasklet: Calculating dense layer output with Tanh activation...\n");

    for (int32_t i = 0; i < 2; i++) {
        for (int32_t j = 0; j < LSTM_HIDDEN_SIZE; j++) {
            dense_output[i] += lstm_output[j] * ((float*)mram_model_weights)[LSTM_HIDDEN_SIZE + i * LSTM_HIDDEN_SIZE + j];
        }
        // Tanh activation using approximation
        dense_output[i] = tanh_approx(dense_output[i]);
    }

    // Softmax计算和标签确定
    float sum_exp = exp_approx(dense_output[0]) + exp_approx(dense_output[1]);
    float prob_0 = exp_approx(dense_output[0]) / sum_exp;
    float prob_1 = exp_approx(dense_output[1]) / sum_exp;

    // 验证Softmax输出
    printf("Tasklet: Probabilities: %f (Class 0), %f (Class 1)\n", prob_0, prob_1);

    // 将分类结果设置为标签
    if (prob_1 > prob_0) {
        example->label = 1; // 恶意地址
    } else {
        example->label = 0; // 非恶意地址
    }

    // 验证最终标签
    printf("Tasklet: Predicted label: %d\n", example->label);

    // 释放动态分配的内存
    mem_reset();  // 使用 mem_reset 释放分配的内存
}



// 预处理域名函数
void preprocess_domain(char *domain, int *char_ids, int max_len) {
    int len = strlen(domain);
    for (int i = 0; i < len; i++) {
        char_ids[i] = mram_vocabulary[(int)domain[i]];
    }
    for (int i = len; i < max_len; i++) {
        char_ids[i] = 0;
    }
}

int main() {
    struct dpu_set_t set, dpu;
    DPU_ASSERT(dpu_alloc(1, NULL, &set));

    int vocab[MAX_DOMAIN_LEN];
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, vocab));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_MRAM, 0, vocab, sizeof(vocab), DPU_XFER_DEFAULT));

    char *domains_need_predict[] = {
        "baidu.com", "01ol.ee4kdushuba.com", "nhy655.3322.org",
        "tmall.com", "office.com", "taobao.com", "dropbox.com"
    };
    nr_domains = sizeof(domains_need_predict) / sizeof(domains_need_predict[0]);

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, domains_need_predict));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_MRAM, 0, domains_need_predict, sizeof(domains_need_predict), DPU_XFER_DEFAULT));

    // 加载模型权重到MRAM
    load_model_weights_to_mram(set);

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
    }

    Example results[NR_TASKLETS];
    DPU_FOREACH(set, dpu) {
        __mram_ptr void* results_mram_addr = (__mram_ptr void*)0x00000000;

        int32_t results_wram_buffer[sizeof(results)/sizeof(int32_t)];
        size_t offset = 0;
        while (offset < sizeof(results)) {
            size_t chunk_size = (sizeof(results) - offset > 2048) ? 2048 : (sizeof(results) - offset);
            mram_read((__mram_ptr void*)((uintptr_t)results_mram_addr + offset), (void*)((uintptr_t)results_wram_buffer + offset), chunk_size);
            offset += chunk_size;
        }

        memcpy(results, results_wram_buffer, sizeof(results));
    }

    for (int i = 0; i < NR_TASKLETS; i++) {
        int label = results[i].label;
        if (label == 1) {
            printf("%s 检测结果为 : 恶意地址\n", results[i].domain_name);
        } else {
            printf("%s 检测结果为 : 非恶意地址\n", results[i].domain_name);
        }
    }

    DPU_ASSERT(dpu_free(set));
    return 0;
}


// 更新MRAM读写操作以处理块和8字节倍数限制
void read_example_from_mram(__mram_ptr void* mram_addr, Example* example) {
    size_t offset = 0;
    size_t chunk_size = sizeof(Example);
    while (offset < chunk_size) {
        size_t read_size = (chunk_size - offset > 2048) ? 2048 : (chunk_size - offset);
        if (read_size % 8 != 0) {
            read_size = (read_size / 8) * 8; // 确保为8的倍数
        }
        mram_read((__mram_ptr void*)((uintptr_t)mram_addr + offset), (void*)((uintptr_t)example + offset), read_size);
        offset += read_size;
    }
}

void write_example_to_mram(Example* example, __mram_ptr void* mram_addr) {
    size_t offset = 0;
    size_t chunk_size = sizeof(Example);
    while (offset < chunk_size) {
        size_t write_size = (chunk_size - offset > 2048) ? 2048 : (chunk_size - offset);
        if (write_size % 8 != 0) {
            write_size = (write_size / 8) * 8; // 确保为8的倍数
        }
        mram_write((void*)((uintptr_t)example + offset), (__mram_ptr void*)((uintptr_t)mram_addr + offset), write_size);
        offset += write_size;
    }
}


// 更新DPU侧任务函数
void dpu_task() {
    unsigned int tasklet_id = me();

    Example *examples = (Example *)mem_alloc(MAX_DOMAIN_LEN * sizeof(Example));
    if (examples == NULL) {
        printf("Tasklet %d: Failed to allocate memory for examples\n", tasklet_id);
        return;
    }
    printf("Tasklet %d: Allocated memory at WRAM address %p\n", tasklet_id, examples);

    for (unsigned int i = tasklet_id; i < nr_domains; i += NR_TASKLETS) {
        __mram_ptr void *domain_mram_addr = (__mram_ptr void *)&domains[i * MAX_DOMAIN_LEN];
        
        // 读取数据到 WRAM
        size_t offset = 0;
        while (offset < sizeof(Example)) {
            size_t chunk_size = (sizeof(Example) - offset > 2048) ? 2048 : (sizeof(Example) - offset);
            mram_read((__mram_ptr void*)((uintptr_t)domain_mram_addr + offset), (void*)((uintptr_t)examples + offset), chunk_size);
            offset += chunk_size;
        }

        printf("Tasklet %d: Read domain name %s (length %d) from MRAM\n", tasklet_id, examples->domain_name, examples->domain_len);

        preprocess_domain(examples[tasklet_id].domain_name, examples[tasklet_id].char_ids, MAX_DOMAIN_LEN);

        printf("Tasklet %d: Preprocessed domain to char IDs: ", tasklet_id);
        for (int j = 0; j < examples[tasklet_id].domain_len && j < 5; j++) {
            printf("%d ", examples[tasklet_id].char_ids[j]);
        }
        printf("\n");

        sem_take(&my_semaphore);
        printf("Tasklet %d: Starting inference...\n", tasklet_id);

        run_inference(&examples[tasklet_id]);

        printf("Tasklet %d: Inference completed. Predicted label: %d\n", tasklet_id, examples[tasklet_id].label);

        sem_give(&my_semaphore);

        // 将结果写回 MRAM
        offset = 0;
        while (offset < sizeof(Example)) {
            size_t chunk_size = (sizeof(Example) - offset > 2048) ? 2048 : (sizeof(Example) - offset);
            mram_write((void*)((uintptr_t)examples + offset), (__mram_ptr void*)((uintptr_t)domain_mram_addr + offset), chunk_size);
            offset += chunk_size;
        }

        printf("Tasklet %d: Wrote result back to MRAM for domain %s\n", tasklet_id, examples->domain_name);
    }

    barrier_wait(&my_barrier);
    printf("Tasklet %d: All tasks completed. Freeing allocated memory.\n", tasklet_id);
    
    mem_reset();  // 释放所有动态分配的内存
}
