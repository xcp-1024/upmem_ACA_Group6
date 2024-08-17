#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define DPU_BINARY "./dt1"  // 定义 DPU 二进制文件的名称
#define NR_TASKLETS 7  // 与DPU任务数一致
#define MAX_DOMAIN_LEN 32  // 最大域名长度

typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];  // 添加了用于存储字符ID的数组
    int32_t label;  // 新添加的label字段
    int32_t padding[2];  // 用于8字节对齐
} Example;

void transfer_examples_to_dpu(struct dpu_set_t set, Example *examples) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_to(dpu, "mram_examples", 0, examples, NR_TASKLETS * sizeof(Example)));
    }
}

void read_results_from_dpu(struct dpu_set_t set, Example *results) {
    struct dpu_set_t dpu;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "mram_examples", 0, results, NR_TASKLETS * sizeof(Example)));
    }
}

int main() {
    struct dpu_set_t set;
    struct dpu_set_t dpu;

    printf("Allocating DPU set...\n");
    DPU_ASSERT(dpu_alloc(1, "backend=simulator", &set));
    printf("DPU set allocated successfully.\n");

    printf("Loading DPU binary: %s\n", DPU_BINARY);
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    // 初始化 Example 数据
    Example examples[NR_TASKLETS] = {
        { .domain_len = 10, .domain_name = "baidu.com" },
        { .domain_len = 21, .domain_name = "01ol.ee4kdushuba.com" },
        { .domain_len = 16, .domain_name = "nhy655.3322.org" },
        { .domain_len = 10, .domain_name = "tmall.com" },
        { .domain_len = 11, .domain_name = "office.com" },
        { .domain_len = 11, .domain_name = "taobao.com" },
        { .domain_len = 12, .domain_name = "dropbox.com" },
    };

    // 传输 Example 数据到 DPU
    printf("Transferring examples to DPU MRAM...\n");
    transfer_examples_to_dpu(set, examples);

    // 运行 DPU 程序
    printf("Launching DPU program...\n");
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
    }

    // 从 DPU 读取结果
    Example results[NR_TASKLETS];
    printf("Reading results from DPU MRAM...\n");
    read_results_from_dpu(set, results);

    // 打印结果
    printf("Results:\n");
    for (int i = 0; i < NR_TASKLETS; i++) {
        printf("Domain: %s, Length: %d, Label: %d\n", results[i].domain_name, results[i].domain_len, results[i].label);
    }

    printf("Freeing DPU set...\n");
    DPU_ASSERT(dpu_free(set));

    return 0;
}
