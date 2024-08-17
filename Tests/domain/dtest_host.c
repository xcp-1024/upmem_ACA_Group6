#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  
#include <dpu_management.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./dgaa"  
#endif

#define MAX_DOMAIN_LEN 32  // Ensure this matches the DPU side
#define VOCAB_SIZE (MAX_DOMAIN_LEN * sizeof(int32_t))  // 256 bytes  
#define MODEL_SIZE 1814336
#define NR_TASKLETS 7  // 调整为7个Tasklet
#define DOMAIN_MEM_SIZE 1024  
#define RESULTS_SIZE (NR_TASKLETS * sizeof(Example))  // 调整为7个Tasklet

typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t label;
    int32_t padding[6]; // Ensure structure size is a multiple of 8
} Example;

void check_allocation(struct dpu_set_t *set) {
    uint32_t nr_of_dpus;
    DPU_ASSERT(dpu_get_nr_dpus(*set, &nr_of_dpus));
    if (nr_of_dpus == 0) {
        printf("DPU allocation failed.\n");
        exit(EXIT_FAILURE);
    }
}

void check_alignment(size_t offset, size_t size) {
    if (offset % 8 != 0 || size % 8 != 0) {
        printf("ERROR: Offset or size is not 8-byte aligned! Offset: %zu, Size: %zu\n", offset, size);
        exit(EXIT_FAILURE); 
    }
}

void transfer_vocabulary(struct dpu_set_t set, int *vocab) {
    struct dpu_set_t dpu;
    size_t vocab_size = VOCAB_SIZE;  // Ensure this is 256 bytes or less

    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        while (offset < vocab_size) {
            size_t chunk_size = (vocab_size - offset > 256) ? 256 : (vocab_size - offset);
            if (chunk_size % 8 != 0) {
                chunk_size -= (chunk_size % 8); // Ensure chunk_size is 8-byte aligned
            }
            check_alignment(offset, chunk_size);
            DPU_ASSERT(dpu_copy_to(dpu, "mram_vocabulary", offset, (void*)((uintptr_t)vocab + offset), chunk_size));
            offset += chunk_size;
        }
    }
    printf("Vocabulary transferred successfully.\n");
}


void transfer_domains(struct dpu_set_t set, char **domains, size_t num_domains) {
    struct dpu_set_t dpu;
    size_t total_domain_chars_size = 0;

    for (size_t i = 0; i < num_domains; i++) {
        size_t domain_len = strlen(domains[i]) + 1;
        size_t aligned_len = domain_len;

        if (aligned_len % 8 != 0) {
            aligned_len += (8 - (aligned_len % 8));
        }

        total_domain_chars_size += aligned_len;
    }

    if (total_domain_chars_size % 8 != 0) {
        total_domain_chars_size += (8 - (total_domain_chars_size % 8));
    }

    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        for (size_t i = 0; i < num_domains; i++) {
            size_t domain_len = strlen(domains[i]) + 1;
            size_t aligned_len = domain_len;

            if (aligned_len % 8 != 0) {
                aligned_len += (8 - (aligned_len % 8));
            }

            char aligned_domain[MAX_DOMAIN_LEN] = {0};
            strncpy(aligned_domain, domains[i], domain_len);

            check_alignment(offset, aligned_len);
            DPU_ASSERT(dpu_copy_to(dpu, "mram_domains", offset, aligned_domain, aligned_len));
            offset += aligned_len;
        }
    }
    printf("Domain names transferred successfully.\n");
}

void load_model_weights_to_mram(struct dpu_set_t set, void *model_data) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        while (offset < MODEL_SIZE) {
            size_t chunk_size = (MODEL_SIZE - offset > 2048) ? 2048 : (MODEL_SIZE - offset);
            if (chunk_size % 8 != 0) {
                chunk_size -= (chunk_size % 8);
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

    int vocab[MAX_DOMAIN_LEN] = {0};

    printf("Transferring vocabulary to DPU MRAM...\n");
    transfer_vocabulary(set, vocab);

    char *domains_need_predict[] = {
        "baidu.com", "01ol.ee4kdushuba.com", "nhy655.3322.org",
        "tmall.com", "office.com", "taobao.com", "dropbox.com"
    };
    size_t num_domains = sizeof(domains_need_predict) / sizeof(domains_need_predict[0]);
    printf("Transferring domain names to DPU MRAM...\n");
    transfer_domains(set, domains_need_predict, num_domains);

    printf("Loading model weights from file...\n");
    FILE *model_file = fopen("/mnt/g/upmem/upmem/upmem_ACA-main/model_saved/pytorch_model_1.bin", "rb");
    if (!model_file) {
        perror("Failed to open model weights file");
        exit(EXIT_FAILURE);
    }

    void *model_data_unaligned = malloc(MODEL_SIZE + 8);
    if (!model_data_unaligned) {
        perror("Failed to allocate memory for model data");
        fclose(model_file);
        exit(EXIT_FAILURE);
    }

    void *model_data = (void*)(((uintptr_t)model_data_unaligned + 7) & ~((uintptr_t)7));

    size_t read_size = fread(model_data, 1, MODEL_SIZE, model_file);
    if (read_size != MODEL_SIZE) {
        perror("Failed to read the complete model data");
        free(model_data_unaligned);
        fclose(model_file);
        exit(EXIT_FAILURE);
    }
    fclose(model_file);
    printf("Model weights loaded from file successfully.\n");

    printf("Transferring model weights to DPU MRAM...\n");
    load_model_weights_to_mram(set, model_data);
    free(model_data_unaligned);

    printf("Launching DPU program...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
    }
    printf("DPU program launched successfully.\n");

    Example results[NR_TASKLETS];
    printf("Reading results from DPU MRAM...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "mram_examples", 0, results, RESULTS_SIZE));
    }
    printf("Results read successfully.\n");

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
