#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_DOMAIN_LEN 120

#define EXTRA_DATA_LEN 6 // 可以逐步增加这个值，直到找到问题点

typedef struct __attribute__((aligned(8))) {
    int32_t domain_len;
    char domain_name[MAX_DOMAIN_LEN];
    int32_t char_ids[MAX_DOMAIN_LEN];
    int32_t extra_data[EXTRA_DATA_LEN];
    int32_t label;
} Example;

__mram_noinit Example mram_example;

void dpu_task() {
    Example example;
    Example read_example;

    // 初始化 Example 结构体
    example.domain_len = strlen("domain.com");
    strncpy(example.domain_name, "domain.com", MAX_DOMAIN_LEN);

    // 填充 char_ids 数组
    for (int i = 0; i < example.domain_len; i++) {
        example.char_ids[i] = (int)example.domain_name[i];
    }

    // 填充 extra_data 数组
    for (int i = 0; i < EXTRA_DATA_LEN; i++) {
        example.extra_data[i] = i * 10;
    }

    // 设置 label
    example.label = 1;

    // 确保对齐到8字节
    size_t aligned_size = (sizeof(Example) + 7) & ~7;

    // 将 Example 结构体写入 MRAM
    mram_write(&example, &mram_example, aligned_size);

    // 使用 mem_reset 释放内存
    mem_reset();

    // 从 MRAM 中读取 Example 结构体
    mram_read(&mram_example, &read_example, aligned_size);

    // 打印读取到的字段
    printf("Read domain_len from MRAM: %d\n", read_example.domain_len);
    printf("Read domain_name from MRAM: %s\n", read_example.domain_name);

    for (int i = 0; i < read_example.domain_len; i++) {
        printf("Read char_ids[%d] from MRAM: %d\n", i, read_example.char_ids[i]);
    }

    for (int i = 0; i < EXTRA_DATA_LEN; i++) {
        printf("Read extra_data[%d] from MRAM: %d\n", i, read_example.extra_data[i]);
    }

    printf("Read label from MRAM: %d\n", read_example.label);

    // 使用 mem_reset 释放内存
    mem_reset();
}

int main() {
    dpu_task();
    return 0;
}
