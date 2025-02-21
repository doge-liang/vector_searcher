#include <faiss/IndexFlat.h>
#include <cstdio>
#include <cstdlib>

int main() {
    int d = 64;      // 向量维度
    int nb = 100000; // 数据库大小
    int nq = 5;      // 查询数量

    // 生成随机数据
    float* xb = new float[d * nb];
    float* xq = new float[d * nq];
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) xb[d * i + j] = drand48();
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) xq[d * i + j] = drand48();
    }

    // 创建索引
    faiss::IndexFlatL2 index(d); // 使用L2距离（欧氏距离）
    printf("Is trained: %s\n", index.is_trained ? "true" : "false");
    
    // 添加数据到索引
    index.add(nb, xb);
    printf("Index total: %ld\n", index.ntotal);

    // 搜索最近邻
    int k = 4; // 返回最近邻数量
    faiss::idx_t* I = new faiss::idx_t[nq * k];
    float* D = new float[nq * k];

    index.search(nq, xq, k, D, I);

    // 打印结果
    for (int i = 0; i < nq; i++) {
        printf("Query %d results:\n", i);
        for (int j = 0; j < k; j++) {
            printf("  %5ld (distance %.3f)\n", I[i * k + j], D[i * k + j]);
        }
    }

    // 清理内存
    delete[] xb;
    delete[] xq;
    delete[] I;
    delete[] D;

    return 0;
}
