# CenterNet
CenterNet for object detection


为什么不用FPN？

FPN是一种分而治之的方式，解决多尺度预测的问题，不同尺度相当于不同的焦距，从而聚焦于不同大小的物体上。
而CenterNet则是合而治之的方式，我们通过这么大的feature map去回归，既能有小物体，也能有大物体，所有物体的尺度信息都在一个feature map上就OK了，
这和TridentNet有异曲同工之妙，只不过，TridentNet最后还是分支了。
在没有一个可靠的理论解释前，实验结果是有很大说服力的，从CenterNet在COCO上的表现就可以看出，这种合而治之的方式完全不比FPN的分而治之的方式要差，整体的框架还更加简洁。

总结：不知道为什么但实验结果可以

为什么没有NMS?
3成3 的maxpooling核去在输出的map上筛选出全是峰值点的map，
然后拿这个map和输出的map做对应，就只有峰值点的map了，
之所以这么做，是因为筛选的时候，步长为1，
那么同一个峰值点会出现在不同的地方，因此，筛选出来的是用不了的，得和原来的输出的map做一下对应才行

总结：加上没有坏处，没法容忍重复框，建议加上NMS


https://www.cnblogs.com/silence-cho/p/13955766.html