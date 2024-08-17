#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <alloc.h>
#include <sem.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>  // 加入stdio.h头文件
#include <stddef.h>  // 包含 offsetof 宏

#define NR_TASKLETS 11
#define MAX_DOMAIN_LEN 164
#define EMBEDDING_DIM 64
#define LSTM_HIDDEN_SIZE 100
#define LSTM_LAYERS 2
#define BIDIRECTIONAL 1
#define MODEL_SIZE 1814336
#define DOMAIN_MEM_SIZE 1024

BARRIER_INIT(my_barrier, NR_TASKLETS);
SEMAPHORE_INIT(my_semaphore, 1);

typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;                 // 偏移量 0, 对齐
    int32_t padding1;                   // 填充到 8 字节对齐
    char domain_name[MAX_DOMAIN_LEN];   // 偏移量 8, 对齐
    int32_t padding2;                   // 填充到 8 字节对齐
    int32_t char_ids[MAX_DOMAIN_LEN];   // 偏移量 176, 对齐
    int32_t label;                      // 偏移量 832, 对齐
    int32_t padding3[4];                // 确保结构体大小为8的倍数，避免对齐问题
} Example;



/*void adjust_padding() {
    size_t label_offset = offsetof(Example, label);
    size_t next_aligned_offset = (label_offset + 7) & ~7;  // 计算下一个8字节对齐的偏移量
    size_t padding_size = next_aligned_offset - label_offset;

    printf("Label offset: %zu, Padding size: %zu\n", label_offset, padding_size);

    // 检查对齐情况
    if (padding_size > 0) {
        printf("Warning: label is not aligned. Manual adjustment required.\n");
        // 此时，你可以打印提示信息并手动调整结构体中的填充字段
    }
}*/







__mram_noinit int32_t mram_vocabulary[MAX_DOMAIN_LEN];
__mram_noinit float mram_model_weights[MODEL_SIZE / sizeof(float)];
__mram_noinit Example mram_examples[NR_TASKLETS];
__mram_noinit char mram_domains[DOMAIN_MEM_SIZE];

// 定义 exp_approx 函数
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

void preprocess_domain(char *domain, int *char_ids, int max_len) {
    int len = strlen(domain);
    for (int i = 0; i < len; i++) {
        char_ids[i] = mram_vocabulary[(int)domain[i]];
    }
    for (int i = len; i < max_len; i++) {
        char_ids[i] = 0;
    }
}

void run_inference(Example *example) {
    // 动态分配内存来存储嵌入矩阵，注意这里的大小
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

    for (int32_t i = 0; i < 2; i++) {
        for (int32_t j = 0; j < LSTM_HIDDEN_SIZE; j++) {
            dense_output[i] += lstm_output[j] * ((float*)mram_model_weights)[LSTM_HIDDEN_SIZE + i * LSTM_HIDDEN_SIZE + j];
        }
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

    Example example;
    __mram_ptr void *example_mram_addr = (__mram_ptr void *)(&mram_examples[tasklet_id]);

    //检查每个字段的偏移量和大小是否为8字节对齐
    size_t domain_len_offset = offsetof(Example, domain_len);
    size_t domain_name_offset = offsetof(Example, domain_name);
    size_t char_ids_offset = offsetof(Example, char_ids);
    size_t label_offset = offsetof(Example, label);

    if (tasklet_id == 0) {
        printf("domain_len offset: %zu, aligned: %s\n", domain_len_offset, (domain_len_offset % 8 == 0) ? "yes" : "no");
        printf("domain_name offset: %zu, aligned: %s\n", domain_name_offset, (domain_name_offset % 8 == 0) ? "yes" : "no");
        printf("char_ids offset: %zu, aligned: %s\n", char_ids_offset, (char_ids_offset % 8 == 0) ? "yes" : "no");
        printf("label offset: %zu, aligned: %s\n", label_offset, (label_offset % 8 == 0) ? "yes" : "no");

        printf("Size of Example struct: %zu\n", sizeof(Example));
    }

    // 从MRAM读取Example结构
    mram_read(example_mram_addr, &example, sizeof(Example));


    // 验证从MRAM读取的数据
    if (tasklet_id == 0) {
        printf("Tasklet %d: Read domain_len: %d\n", tasklet_id, example.domain_len);
        printf("Tasklet %d: Read domain_name: %s\n", tasklet_id, example.domain_name);
        printf("Tasklet %d: Read first char_id: %d\n", tasklet_id, example.char_ids[0]);

        // 打印所有的 char_ids，检查对齐问题
        for (int i = 0; i < 10; i++) {
            printf("Tasklet %d: char_ids[%d] = %d\n", tasklet_id, i, example.char_ids[i]);
        }
    }

    sem_take(&my_semaphore);
    run_inference(&example);
    sem_give(&my_semaphore);

    mram_write(&example, example_mram_addr, sizeof(Example));

    barrier_wait(&my_barrier);

    /*if (tasklet_id == 0) {
        printf("Tasklet %d: Inference result - label: %d\n", tasklet_id, example.label);
    }*/
}





int main() {
    //adjust_padding();
    dpu_task();
    return 0;
}


