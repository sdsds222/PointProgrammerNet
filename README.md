# PointProgrammerNet
PointProgrammerNet 将无序点集写入固定尺寸的 Fast Weight Program（FWP）矩阵，再用任意空间坐标查询局部统计。它不依赖 FPS、KNN 或离散体素层级；点的写入可以求和，因此同一套表示既能处理静态点云，也能自然扩展到持续到来的点云帧。
