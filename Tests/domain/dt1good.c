#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <alloc.h>  
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define NR_TASKLETS 4
#define MAX_DOMAIN_LEN 16
#define EMBEDDING_DIM 4
#define LSTM_HIDDEN_DIM 4  // 隐藏层维度

BARRIER_INIT(my_barrier, NR_TASKLETS);

typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t label;
    float padding[EMBEDDING_DIM * 2];  
    float lstm_hidden_1[LSTM_HIDDEN_DIM];  
    float lstm_hidden_2[LSTM_HIDDEN_DIM];  
} Example;

__mram_noinit Example mram_examples[NR_TASKLETS];
__mram_noinit float mram_embedding_matrix[256 * EMBEDDING_DIM]; 
__mram_noinit float lstm_weights_1[EMBEDDING_DIM * LSTM_HIDDEN_DIM];  
__mram_noinit float lstm_weights_2[LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM];  

void dpu_task() {
    unsigned int tasklet_id = me(); 

    Example example;

    // 从MRAM读取域名和标签
    __mram_ptr void *example_mram_addr = (__mram_ptr void *)(&mram_examples[tasklet_id]);
    mram_read(example_mram_addr, &example, sizeof(Example));

    // 初始化LSTM隐藏状态
    for (int j = 0; j < LSTM_HIDDEN_DIM; j++) {
        example.lstm_hidden_1[j] = 0.0f;
        example.lstm_hidden_2[j] = 0.0f;
    }

    // LSTM第1层处理
    for (int i = 0; i < example.domain_len; i++) {
        int char_id = example.char_ids[i];
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            example.padding[j] = mram_embedding_matrix[char_id * EMBEDDING_DIM + j];
        }
        for (int j = 0; j < LSTM_HIDDEN_DIM; j++) {
            for (int k = 0; k < EMBEDDING_DIM; k++) {
                example.lstm_hidden_1[j] += example.padding[k] * lstm_weights_1[k * LSTM_HIDDEN_DIM + j];
            }
        }
    }

    // LSTM第2层处理
    for (int i = 0; i < LSTM_HIDDEN_DIM; i++) {
        for (int j = 0; j < LSTM_HIDDEN_DIM; j++) {
            for (int k = 0; k < LSTM_HIDDEN_DIM; k++) {
                example.lstm_hidden_2[j] += example.lstm_hidden_1[k] * lstm_weights_2[k * LSTM_HIDDEN_DIM + j];
            }
        }
    }

    // 应用输出门
    float output_gate = 0.8f;
    for (int j = 0; j < LSTM_HIDDEN_DIM; j++) {
        example.lstm_hidden_2[j] *= output_gate;
    }

    // 存储最终结果
    float final_output = 0.0f;
    for (int j = 0; j < LSTM_HIDDEN_DIM; j++) {
        final_output += example.lstm_hidden_2[j];
    }
    example.padding[EMBEDDING_DIM] = final_output;

    // 将结果写回MRAM
    mram_write(&example, example_mram_addr, sizeof(Example));

    // 等待所有tasklet完成
    barrier_wait(&my_barrier);
}

int main() {
    dpu_task();
    return 0;
}
