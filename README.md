# Nesting
Irregular packing using NFP and GA

# Naive Algorithm
计算每个图形的最小包络矩形。通过遍历每个零件的具体坐标找到最边缘的坐标点，从而计算出最小包络矩形，计算的过程中考虑相邻零件的最小间隔。得到各零件的最小包络矩形之后，我们就将进行各零件的排样。这里采用的是以最小包络矩形的宽度降序进行排样。整个流程大致为：首先选取宽度最大的矩形将其放在布料的左下角，然后以其宽划分布料，再将次宽度矩形排列再其上方，依次重复次操作。

# NFP and GA
可参考[2][3][4][5],这部分代码重构完再放上来。
# Reference
[1] 	阿里云天池大赛. 2019广东工业智造创新大赛【赛场二】[EB/OL]. https://tianchi.aliyun.com/competition/entrance/231749/information, 2019–08–19/2019–12–17.

[2] 	Jack000. SVGnest [EB/OL]. https://github.com/Jack000/SVGnest, 2019–04–11/2019–12–17.

[3] 	auto. 遗传算法（python版）[EB/OL]. http://www.py3study.com/Article/details/id/18603.html, 2019–09–25/2019–12–17.

[4] 	Leao A A S, Toledo F M B, Oliveira J F, et al. Irregular packing problems: a review of mathematical models[J]. European Journal of Operational Research, 2019.

[5] 	liangxuCHEN. no_fit_polygon [EB/OL]. https://github.com/liangxuCHEN/no_fit_polygon, 2019–03–19/2019–12–17.

# To do 
1.用Numpy重构代码;

2.考虑瑕点问题;

3.写中英文两版readme.
