#include <defs.h>        // include the definition of me()
#include <barrier.h>
#include <mutex.h>
#include <sem.h>
#include <mram.h>
#include <alloc.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#define NR_TASKLETS 11
#define MAX_DOMAIN_LEN 130
#define EMBEDDING_DIM 128
#define LSTM_HIDDEN_SIZE 100
#define LSTM_LAYERS 2
#define BIDIRECTIONAL 1
#define MODEL_SIZE 1814336

BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(my_mutex);
SEMAPHORE_INIT(my_semaphore, 1);

typedef struct {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t label;
    int32_t padding[6]; // the size of structure must be the fachs of 8
} Example;

__mram_noinit int32_t mram_vocabulary[MAX_DOMAIN_LEN];
__mram_noinit float mram_model_weights[MODEL_SIZE / sizeof(float)];
__mram_noinit Example mram_examples[NR_TASKLETS];

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
    float embeddings[MAX_DOMAIN_LEN][EMBEDDING_DIM];
    printf("Domain length: %d\n", example->domain_len);

    for (int32_t i = 0; i < example->domain_len; i++) {
        int32_t char_id = example->char_ids[i];
        for (int32_t j = 0; j < EMBEDDING_DIM; j++) {
            embeddings[i][j] = ((float*)mram_model_weights)[char_id * EMBEDDING_DIM + j];
        }
    }

    float lstm_output[LSTM_HIDDEN_SIZE] = {0};
    float lstm_backward_output[LSTM_HIDDEN_SIZE] = {0};

    for (int32_t layer = 0; layer < LSTM_LAYERS; layer++) {
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

    float dense_output[2] = {0};

    for (int32_t i = 0; i < 2; i++) {
        for (int32_t j = 0; j < LSTM_HIDDEN_SIZE; j++) {
            dense_output[i] += lstm_output[j] * ((float*)mram_model_weights)[LSTM_HIDDEN_SIZE + i * LSTM_HIDDEN_SIZE + j];
        }
        dense_output[i] = tanh_approx(dense_output[i]);
    }

    float sum_exp = exp_approx(dense_output[0]) + exp_approx(dense_output[1]);
    float prob_0 = exp_approx(dense_output[0]) / sum_exp;
    float prob_1 = exp_approx(dense_output[1]) / sum_exp;

    if (prob_1 > prob_0) {
        example->label = 1;
    } else {
        example->label = 0;
    }
}

void dpu_task() {
    unsigned int tasklet_id = me();  // get the id of tasklet

    Example example;
    __mram_ptr void *example_mram_addr = (__mram_ptr void *)(&mram_examples[tasklet_id]);

    // divided the procesing of reading structure in 8
    for (int offset = 0; offset < sizeof(Example); offset += 8) {
        mram_read((__mram_ptr void *)((uintptr_t)example_mram_addr + offset), (void *)((uintptr_t)&example + offset), 8);
    }

    // 打印读取的 Example 结构体内容
    printf("Tasklet %u: example.domain_len = %d, example.domain_name = %s\n", 
            tasklet_id, example.domain_len, example.domain_name);

    preprocess_domain(example.domain_name, example.char_ids, MAX_DOMAIN_LEN);

    sem_take(&my_semaphore);
    run_inference(&example);
    sem_give(&my_semaphore);

    // divided the procesing of writing structure in 8
    for (int offset = 0; offset < sizeof(Example); offset += 8) {
        mram_write((void *)((uintptr_t)&example + offset), (__mram_ptr void *)((uintptr_t)example_mram_addr + offset), 8);
    }

    barrier_wait(&my_barrier);
}

int main() {
    dpu_task();
    return 0;
}
