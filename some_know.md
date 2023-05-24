1. mask理解
https://www.cvmart.net/community/detail/4493
key_padding_mask的作用
这里举一个简单的例子：

现在有一个batch，batch_size = 3，长度为4，token表现形式如下：
```
[
    [‘a’,'b','c','<PAD>'],
    [‘a’,'b','c','d'],
    [‘a’,'b','<PAD>','<PAD>']
]
```
现在假设你要对其进行self-attention的计算（可以在encoder，也可以在decoder），那么以第三行数据为例，‘a’在做qkv计算的时候，会看到'b','<PAD>','<PAD>'，但是我们不希望‘a’看到'<PAD>'，因为他们本身毫无意义，所以，需要key_padding_mask遮住他们。

key_padding_mask的形状大小为（N,S），对应这个例子，key_padding_mask为以下形式，key_padding_mask.shape = （3,4）：
```
[
    [False, False, False, True],
    [False, False, False, False],
    [False, False, True, True]
]
```
值得说明的是，key_padding_mask本质上是遮住key这个位置的值（置0），但是<PAD> token本身，也是会做qkv的计算的，以第三行数据的第三个位置为例，它的q是<PAD>的embedding，k和v分别各是第一个的‘a’和第二个的‘b’，它也会输出一个embedding。

所以你的模型训练在transformer最后的output计算loss的时候，还需要指定ignoreindex=pad_index。以第三行数据为例，它的监督信号是[3205,1890,0,0]，pad_index=0 。如此一来，即便位于<PAD>的transformer会疯狂的和有意义的position做qkv，也会输出embedding，但是我们不算它的loss，任凭它各种作妖。
    
总结: 也就是说通过pad补充固定长度后，pad也会做qkv, 但是只要我们将pad对应的pad_index指定为ignoreindex，算loss的时候，不会给他算loss
