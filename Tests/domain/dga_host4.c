#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./dga"  
#endif

#define MAX_DOMAIN_LEN 130
#define MODEL_SIZE 1814336
#define NR_TASKLETS 11

typedef struct {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t label;
    int32_t padding[6]; // Ensure the struct size is a multiple of 8 bytes
} Example;

void check_alignment(size_t offset, size_t size) {
    if (offset % 8 != 0 || size % 8 != 0) {
        printf("ERROR: Offset or size is not 8-byte aligned! Offset: %zu, Size: %zu\n", offset, size);
        exit(EXIT_FAILURE);
    }
    else printf("Offset: %zu, Size: %zu\n", offset, size);
}

// Transfer data in aligned chunks
void transfer_data(struct dpu_set_t set, const char *mram_name, void *data, size_t data_size) {
    struct dpu_set_t dpu;  // Define dpu variable here
    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        while (offset < data_size) {
            size_t chunk_size = (data_size - offset > 2048) ? 2048 : (data_size - offset);

            // Ensure chunk_size is 8-byte aligned
            if (chunk_size % 8 != 0) {
                chunk_size = (chunk_size / 8) * 8;
            }

            // Ensure offset is 8-byte aligned
            if (offset % 8 != 0) {
                size_t alignment_adjustment = 8 - (offset % 8);
                if (alignment_adjustment + offset > data_size) {
                    alignment_adjustment = data_size - offset;
                }
                // Adjust offset
                offset += alignment_adjustment;
                if (offset >= data_size) {
                    break;  // If alignment adjustment goes beyond data_size, break
                }
                continue;  // Retry with adjusted offset
            }

            check_alignment(offset, chunk_size);
            printf("Transferring %s to DPU at offset %zu (chunk size %zu)\n", mram_name, offset, chunk_size);
            DPU_ASSERT(dpu_copy_to(dpu, mram_name, offset, (void*)((uintptr_t)data + offset), chunk_size));
            offset += chunk_size;
        }
    }
    printf("%s transferred successfully.\n", mram_name);
}

int main() {
    struct dpu_set_t set;

    printf("Allocating DPU set...\n");
    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    printf("DPU set allocated successfully.\n");

    int vocab[MAX_DOMAIN_LEN] = {0};
    size_t vocab_size = sizeof(vocab);
    vocab_size = (vocab_size + 7) & ~7;  // Ensure 8-byte alignment
    check_alignment(0, vocab_size);  // Check alignment for vocab_size

    printf("Loading DPU binary: %s\n", DPU_BINARY);
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    printf("DPU binary loaded successfully.\n");

    printf("Transferring vocabulary to DPU MRAM...\n");
    transfer_data(set, "mram_vocabulary", vocab, vocab_size);

    // Prepare domain names buffer
    char domain_names[MAX_DOMAIN_LEN * 7] = {0};  // Assuming each domain name is less than MAX_DOMAIN_LEN
    size_t domains_size = 0;
    char *domains_need_predict[] = {
        "baidu.com", "01ol.ee4kdushuba.com", "nhy655.3322.org",
        "tmall.com", "office.com", "taobao.com", "dropbox.com"
    };
    size_t domains_count = sizeof(domains_need_predict) / sizeof(domains_need_predict[0]);

    // Calculate total size of domain names
    for (size_t i = 0; i < domains_count; ++i) {
        size_t len = strlen(domains_need_predict[i]) + 1;  // +1 for null terminator
        memcpy(domain_names + domains_size, domains_need_predict[i], len);
        domains_size += len;
    }
    
    // Ensure domains_size is 8-byte aligned
    domains_size = (domains_size + 7) & ~7;
    check_alignment(0, domains_size);  // Check alignment for domains_size

    printf("Number of domains to predict: %zu\n", domains_count);
    printf("Transferring domain names to DPU MRAM...\n");
    transfer_data(set, "domains", domain_names, domains_size);

    // Other processing code...

    return 0;
}
