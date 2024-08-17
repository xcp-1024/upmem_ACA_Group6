#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    int32_t padding[6]; // 确保结构体大小为8的倍数
} Example;

// 检查偏移量和大小是否是8字节对齐的
void check_alignment(size_t offset, size_t size) {
    if (offset % 8 != 0 || size % 8 != 0) {
        printf("ERROR: Offset or size is not 8-byte aligned! Offset: %zu, Size: %zu\n", offset, size);
        exit(EXIT_FAILURE); // 如果不对齐，退出程序
    }
    else printf("Offset: %zu, Size: %zu\n", offset, size);
}

void load_model_weights_to_mram(struct dpu_set_t set, void *model_data) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        size_t offset = 0;
        while (offset < MODEL_SIZE) {
            // 调整 chunk_size，确保它是8的倍数
            size_t remaining = MODEL_SIZE - offset;
            size_t chunk_size = (remaining > 2048) ? 2048 : remaining;
            
            // 确保 chunk_size 是 8 的倍数
            if (chunk_size % 8 != 0) {
                chunk_size = (chunk_size / 8) * 8;  // 向下调整为最接近的8的倍数
            }

            // 检查 offset 和 chunk_size 是否符合 8 字节对齐
            check_alignment(offset, chunk_size);

            printf("Copying model weights to DPU at offset %zu (chunk size %zu)\n", offset, chunk_size);
            DPU_ASSERT(dpu_copy_to(dpu, "mram_model_weights", offset, (void*)((uintptr_t)model_data + offset), chunk_size));
            
            offset += chunk_size;
        }
    }
    printf("Model weights loaded into MRAM.\n");
}



int main() {
    struct dpu_set_t set, dpu;
    printf("Allocating DPU set...\n");
    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    printf("DPU set allocated successfully.\n");

        // 调整 vocab 的大小确保 8 字节对齐
    int vocab[MAX_DOMAIN_LEN] = {0};
    size_t vocab_size = sizeof(vocab);

    // 确保 vocab_size 是 8 字节对齐的
    if (vocab_size % 8 != 0) {
        size_t padding = 8 - (vocab_size % 8);
        // 实际增加填充字节到数组
        char padding_bytes[padding];
        memset(padding_bytes, 0, padding);
        vocab_size += padding;

        // 将填充字节追加到传输数据
        printf("Adding padding to vocabulary to ensure 8-byte alignment...\n");
        DPU_FOREACH(set, dpu) {
            DPU_ASSERT(dpu_copy_to(dpu, "mram_vocabulary", 0, vocab, vocab_size - padding));
            DPU_ASSERT(dpu_copy_to(dpu, "mram_vocabulary", vocab_size - padding, padding_bytes, padding));
        }
    } else {
        // 无需填充，直接传输数据
        DPU_FOREACH(set, dpu) {
            DPU_ASSERT(dpu_copy_to(dpu, "mram_vocabulary", 0, vocab, vocab_size));
        }
    }

printf("Vocabulary size after padding (if any): %zu bytes\n", vocab_size);

    // 在 dpu_copy_to 之前检查对齐
    printf("Checking alignment before transferring vocabulary...\n");
    check_alignment(0, vocab_size);  // 这里的 0 是指偏移量为 0

    printf("Loading DPU binary: %s\n", DPU_BINARY);
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    printf("DPU binary loaded successfully.\n");



    printf("Transferring vocabulary to DPU MRAM...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "mram_vocabulary", 0, vocab, vocab_size));
    }
    printf("Vocabulary transferred successfully.\n");

    char *domains_need_predict[] = {
        "baidu.com", "01ol.ee4kdushuba.com", "nhy655.3322.org",
        "tmall.com", "office.com", "taobao.com", "dropbox.com"
    };
    int nr_domains = sizeof(domains_need_predict) / sizeof(domains_need_predict[0]);
    printf("Number of domains to predict: %d\n", nr_domains);

    size_t domains_size = sizeof(domains_need_predict);

    // 确保 domains_size 是 8 字节对齐的
    if (domains_size % 8 != 0) {
        domains_size = ((domains_size / 8) + 1) * 8;
    }

    // 在 dpu_copy_to 之前检查对齐
    printf("Checking alignment before transferring domain names...\n");
    check_alignment(0, domains_size);  // 这里的 0 是指偏移量为 0

    printf("Transferring domain names to DPU MRAM...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "domains", 0, domains_need_predict, domains_size));
    }
    printf("Domain names transferred successfully.\n");

    printf("Loading model weights from file...\n");
    FILE *model_file = fopen("pytorch_model_1.bin", "rb");
    if (!model_file) {
        perror("Failed to open model weights file");
        exit(EXIT_FAILURE);
    }

    void *model_data = malloc(MODEL_SIZE);
    if (!model_data) {
        perror("Failed to allocate memory for model data");
        fclose(model_file);
        exit(EXIT_FAILURE);
    }

    size_t read_size = fread(model_data, 1, MODEL_SIZE, model_file);
    if (read_size != MODEL_SIZE) {
        perror("Failed to read the complete model data");
        free(model_data);
        fclose(model_file);
        exit(EXIT_FAILURE);
    }
    fclose(model_file);
    printf("Model weights loaded from file successfully.\n");

    // 在调用 load_model_weights_to_mram 之前检查并打印对齐信息
    printf("Checking alignment before transferring model weights...\n");
    check_alignment(0, MODEL_SIZE);  // 这里的 0 是指偏移量为 0

    printf("Transferring model weights to DPU MRAM...\n");
    load_model_weights_to_mram(set, model_data);
    free(model_data);

    printf("Launching DPU program...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
    }
    printf("DPU program launched successfully.\n");

    Example results[NR_TASKLETS];
    printf("Reading results from DPU MRAM...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "results", 0, results, sizeof(results)));
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
