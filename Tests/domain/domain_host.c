#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <dpu.h>
#include <dpu_log.h>

#define MAX_DOMAIN_LEN 16
#define EMBEDDING_DIM 4
#define LSTM_HIDDEN_DIM 4
#define VOCAB_SIZE 256
#define NR_TASKLETS 4

float vocab[VOCAB_SIZE];

typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];  // Add this if needed
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t label;
    float padding[EMBEDDING_DIM * 2];  
    float lstm_hidden_1[LSTM_HIDDEN_DIM];  
    float lstm_hidden_2[LSTM_HIDDEN_DIM];  
} Example;

void init_vocab() {
    memset(vocab, 1, sizeof(vocab));  // Default to [UNK]

    vocab['w'] = 2;
    vocab['.'] = 3;
    vocab['u'] = 4;
    vocab['m'] = 5;
    vocab['o'] = 6;
    vocab['f'] = 7;
    vocab['i'] = 8;
    vocab['n'] = 9;
    vocab['v'] = 10;
    vocab['a'] = 11;
    vocab['r'] = 12;
    vocab['z'] = 13;
    vocab['e'] = 14;
    vocab['b'] = 15;
    vocab['s'] = 16;
    vocab['c'] = 17;
    vocab['d'] = 18;
    vocab['h'] = 19;
    vocab['1'] = 20;
    vocab['k'] = 21;
    vocab['t'] = 22;
    vocab['q'] = 23;
    vocab['p'] = 24;
    vocab['0'] = 25;
    vocab['l'] = 26;
    vocab['4'] = 27;
    vocab['y'] = 28;
    vocab['g'] = 29;
    vocab['x'] = 30;
    vocab['5'] = 31;
    vocab['j'] = 32;
    vocab['-'] = 33;
    vocab['3'] = 34;
    vocab['2'] = 35;
    vocab['8'] = 36;
    vocab['7'] = 37;
    vocab['9'] = 38;
    vocab['6'] = 39;
    vocab['Y'] = 40;
    vocab['S'] = 41;
    vocab['D'] = 42;
    vocab['O'] = 43;
    vocab['R'] = 44;
    vocab['G'] = 45;
    vocab['J'] = 46;
    vocab['N'] = 47;
    vocab['C'] = 48;
    vocab['M'] = 49;
    vocab['W'] = 50;
    vocab['Q'] = 51;
    vocab['L'] = 52;
    vocab['_'] = 53;
    vocab['H'] = 54;
    vocab[':'] = 55;
}

void generate_char_ids(const char *domain, int32_t *char_ids, int32_t *domain_len) {
    *domain_len = strlen(domain);
    for (int i = 0; i < *domain_len; i++) {
        char_ids[i] = vocab[(unsigned char)domain[i]];
    }
    for (int i = *domain_len; i < MAX_DOMAIN_LEN; i++) {
        char_ids[i] = 0;
    }
}

void transfer_examples_to_dpu(struct dpu_set_t set, Example *examples) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "mram_examples", 0, examples, NR_TASKLETS * sizeof(Example)));
    }
}

void transfer_embedding_matrix_to_dpu(struct dpu_set_t set, float *embedding_matrix) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "mram_embedding_matrix", 0, embedding_matrix, VOCAB_SIZE * EMBEDDING_DIM * sizeof(float)));
    }
}

void transfer_lstm_weights_to_dpu(struct dpu_set_t set, float *lstm_weights_1, float *lstm_weights_2) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "lstm_weights_1", 0, lstm_weights_1, EMBEDDING_DIM * LSTM_HIDDEN_DIM * sizeof(float)));
        DPU_ASSERT(dpu_copy_to(dpu, "lstm_weights_2", 0, lstm_weights_2, LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM * sizeof(float)));
    }
}

void read_results_from_dpu(struct dpu_set_t set, Example *results) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "mram_examples", 0, results, NR_TASKLETS * sizeof(Example)));
    }
}

void load_weights_from_text(const char *filename, float *lstm_weights_1, float *lstm_weights_2) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    // 修改：直接读取从Python脚本生成的 dpu_lstm_weights_extracted.txt 文件 
    for (int i = 0; i < EMBEDDING_DIM * LSTM_HIDDEN_DIM; i++) {
        if (fscanf(file, "%f", &lstm_weights_1[i]) != 1) {
            printf("Error: Failed to read lstm_weights_1[%d]\n", i);
            fclose(file);
            exit(1);
        }
    }

    for (int i = 0; i < LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM; i++) {
        if (fscanf(file, "%f", &lstm_weights_2[i]) != 1) {
            printf("Error: Failed to read lstm_weights_2[%d]\n", i);
            fclose(file);
            exit(1);
        }
    }

    fclose(file);
}


int main() {
    struct dpu_set_t set;
    struct dpu_set_t dpu;

    // Initialize vocab
    init_vocab();

    // Initialize DPU
    DPU_ASSERT(dpu_alloc(1, "backend=simulator", &set));
    DPU_ASSERT(dpu_load(set, "./dt1", NULL));

    // Initialize domain names and char_ids
    const char *domains[] = {
        "baidu.com",
        "01ol.ee4kdushuba",
        "nhy655.3322.org",
        "tmall.com"
    };

    Example examples[NR_TASKLETS];
    for (int i = 0; i < NR_TASKLETS; i++) {
        strncpy(examples[i].domain_name, domains[i], MAX_DOMAIN_LEN);
        generate_char_ids(domains[i], examples[i].char_ids, &examples[i].domain_len);
    }

    // Initialize embedding matrix
    float embedding_matrix[VOCAB_SIZE * EMBEDDING_DIM];
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            embedding_matrix[i * EMBEDDING_DIM + j] = (float)(i + j);
        }
    }

    // Initialize LSTM weights
    float lstm_weights_1[EMBEDDING_DIM * LSTM_HIDDEN_DIM];
    float lstm_weights_2[LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM];

    // Load actual weights from the extracted text file
    load_weights_from_text("dpu_lstm_weights_extracted.txt", lstm_weights_1, lstm_weights_2);


    // Transfer data to DPU
    transfer_examples_to_dpu(set, examples);
    transfer_embedding_matrix_to_dpu(set, embedding_matrix);
    transfer_lstm_weights_to_dpu(set, lstm_weights_1, lstm_weights_2);

    // Run DPU program
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
    }

    // Read results from DPU
    Example results[NR_TASKLETS];
    read_results_from_dpu(set, results);

    // Print results
    for (int i = 0; i < NR_TASKLETS; i++) {
        printf("Domain: %s\n", results[i].domain_name);
        printf("Char IDs: ");
        for (int j = 0; j < MAX_DOMAIN_LEN; j++) {
            printf("%d ", results[i].char_ids[j]);
        }
        printf("\nEmbedding values: ");
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            printf("%f ", results[i].padding[j]);
        }
        printf("\nLSTM lstm_hidden_1 values: ");
        for (int j = 0; j < LSTM_HIDDEN_DIM; j++) {
            printf("%f ", results[i].lstm_hidden_1[j]);
        }
        printf("\nLSTM lstm_hidden_2 values: ");
        for (int j = 0; j < LSTM_HIDDEN_DIM; j++) {
            printf("%f ", results[i].lstm_hidden_2[j]);
        }
        printf("\nFinal output: %f\n\n", results[i].padding[EMBEDDING_DIM]);
        
    // Assuming a threshold of 10.0 for malicious domain detection
    float threshold = 5.0;
    if (results[i].padding[EMBEDDING_DIM] > threshold) {
        printf("malicious domain\n");
    } else {
        printf("no malicious domain\n");
    }
    printf("\n");
    

}


    // Free DPU resources
    DPU_ASSERT(dpu_free(set));

    return 0;
}
