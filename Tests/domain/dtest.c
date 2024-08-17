#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <alloc.h>
#include <sem.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define NR_TASKLETS 7  // Tasklet数量与域名数量匹配
#define MAX_DOMAIN_LEN 32
#define EMBEDDING_DIM 64
#define LSTM_HIDDEN_SIZE 100
#define LSTM_LAYERS 2
#define BIDIRECTIONAL 1
#define MODEL_SIZE 1814336
#define DOMAIN_MEM_SIZE 1024

BARRIER_INIT(my_barrier, NR_TASKLETS);
SEMAPHORE_INIT(my_semaphore, 1);

typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t label;
    int32_t padding[6]; // 确保结构体大小是8字节的倍数
} Example;

__mram_noinit int32_t mram_vocabulary[MAX_DOMAIN_LEN];
__mram_noinit float mram_model_weights[MODEL_SIZE / sizeof(float)];
__mram_noinit Example mram_examples[NR_TASKLETS];
__mram_noinit char mram_domains[DOMAIN_MEM_SIZE];  // 存储域名字符

void preprocess_domain(char *domain, int *char_ids, int max_len) {
    int len = strlen(domain);
    for (int i = 0; i < len; i++) {
        char_ids[i] = mram_vocabulary[(int)domain[i]];
    }
    for (int i = len; i < max_len; i++) {
        char_ids[i] = 0;
    }
}

float tanh_approx(float x) {
    if (x < -3.0f) return -1.0f;
    else if (x > 3.0f) return 1.0f;
    else return x * (27.0f + x * x) / (27.0f + 9.0f * x * x);
}

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
    // 动态分配内存以存储嵌入矩阵
    float (*embeddings)[EMBEDDING_DIM] = (float (*)[EMBEDDING_DIM])mem_alloc(MAX_DOMAIN_LEN * EMBEDDING_DIM * sizeof(float));
    if (embeddings == NULL) {
        return;  
    }

    // 填充嵌入矩阵
    for (int32_t i = 0; i < example->domain_len; i++) {
        int32_t char_id = example->char_ids[i];
        for (int32_t j = 0; j < EMBEDDING_DIM; j++) {
            embeddings[i][j] = ((float*)mram_model_weights)[char_id * EMBEDDING_DIM + j];
        }
    }

    float lstm_output[LSTM_HIDDEN_SIZE] = {0};
    float lstm_backward_output[LSTM_HIDDEN_SIZE] = {0};

    // LSTM推理
    for (int32_t layer = 0; layer < LSTM_LAYERS; layer++) {
        for (int32_t i = 0; i < example->domain_len; i++) {
            int32_t reverse_index = example->domain_len - i - 1;
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

    float dense_output[2] = {0};

    // 全连接层推理
    for (int32_t i = 0; i < 2; i++) {
        for (int32_t j = 0; j < LSTM_HIDDEN_SIZE; j++) {
            dense_output[i] += lstm_output[j] * ((float*)mram_model_weights)[LSTM_HIDDEN_SIZE + i * LSTM_HIDDEN_SIZE + j];
        }
        dense_output[i] = tanh_approx(dense_output[i]);
    }

    float sum_exp = exp_approx(dense_output[0]) + exp_approx(dense_output[1]);
    float prob_0 = exp_approx(dense_output[0]) / sum_exp;
    float prob_1 = exp_approx(dense_output[1]) / sum_exp;

    example->label = (prob_1 > prob_0) ? 1 : 0;

    // 推理完成后释放内存
    mem_reset();  
}

void dpu_task() {
    unsigned int tasklet_id = me();

    if (tasklet_id >= NR_TASKLETS) {
        return;  // 超过域名数量的Tasklet不进行操作
    }

    Example example;
    __mram_ptr void *example_mram_addr = (__mram_ptr void *)(&mram_examples[tasklet_id]);

    // 从MRAM中读取域名
    mram_read(example_mram_addr, &example, sizeof(Example));

    // 确认读取到的域名
    printf("Tasklet %d: Received domain_name = %s\n", tasklet_id, example.domain_name);

    // 预处理域名
    preprocess_domain(example.domain_name, example.char_ids, MAX_DOMAIN_LEN);

    // 推理过程（使用信号量确保线程安全）
    sem_take(&my_semaphore);
    run_inference(&example);
    sem_give(&my_semaphore);

    // 写回结果到MRAM
    mram_write(&example, example_mram_addr, sizeof(Example));

    // 等待所有Tasklet完成
    barrier_wait(&my_barrier);
}

int main() {
    dpu_task();
    return 0;
}
