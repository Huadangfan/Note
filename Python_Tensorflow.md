# Python and Tensorflow

[toc]
___

## 1. 换源

### anaconda

*当前环境：win10+anaconda3+python3.x*

- 添加清华源：

  ```cmd
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/

  conda config --set show_channel_urls yes
  ```

- 查看目前的源目录

  ```cmd
  conda config --show
  ```
  
- 删除添加的国内源，恢复默认源

  ```cmd
  conda config --remove-key channels
  ```

### pip install

- 使用豆瓣源：

  ```bash
  sudo pip install [the installed package] -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple
  ```

- No module named 'pip'
  solution:

  ```bash
  > python -m ensurepip
  > python -m pip install --upgrade pip
  ```

___

## 2. Markdown

### 语法

```markdown
# first title
## second title
### third title
...
*demo* 斜体

**demo** 加粗

![describe](link) 超链接

___ 分隔符

- 无序小标题

* 无序小标题
    * 次级无序小标题

1. 有序小标题

`demo` code

&emsp; 缩进标识

```

图片居中：

```markdown

<div align=center>

![](figure's link)
</div>

```

### HTML

```html
<br> <!--换行-->
<font size=3></font> <!--指定字体大小-->
<i></i> <!--斜体-->
<b></b> <!--加粗-->
<img src=".." width="25%" height="25%"> <!--插入图片-->
<center></center> <!--居中-->
```

___

## 3.Python

### 常用

#### `.npz`文件读取与存储

- 读取：

  ```python
  import numpy as np

  meta = np.load(filenames)
  print(meta.files)         # 查看npz文件的项目
  # for example, it has 'data', 'itp', 'its'
  data = meta['data']
  # or
  data = copy(meta['data'])
  itp = meta['itp']
  its = meta['its']
  ```

- 存储：

  ```python
  """
  pick_data.npz: filename
  x_train: subfile
  target: subfile
  """
  np.savez('pick_data.npz',x_train=x_train,target=target)
  # save x_train array as x_train file
  # save target array as target file

  np.save('filename.npy',array) # 普通的数组存储方式
  ```

#### Plot

- `plt.tight_layout()`:
    整理布局作用

    ```python
    import matplotlib.pyplot as plt
    plt.tight_layout()
    ```

- `plt.figure()`:
    控制figure属性

    ```python
    import matplotlib.pyplot as plt
    plt.figure(num=None, figsize=None, dpi=None)
    # num 当前figure编号
    # figsize 控制figure长宽及其比例，输入为(**, **)
    # figsize = (*, *)
    ```

-`obspy plot`:
   利用obspy绘制多道地震图
   ```python
   st.plot(outfile='filename.png', automerge=False, equal_scale=False, size=(800,850))
   # automerge: 同label的数据会自动合并在一张图中
   # equal_scale: 振幅用一个scale
   ```
  
- ERROR
  - Fail to allocate bitmap:
      figure打开太多了，内存溢出，solution:

      ```python
      # demo
      fig = plt.gcf()
      figname = str(n)+".png"
      filename = "pred_figure"
      fig.savefig(os.path.join(filename,figname))
      # there are the 1000+ loop, too many figure are opened
      # return error: Fail to allocate bitmap
      # add this command:
      plt.close('all') #close all figure
      ```

### 注释与缩进

#### `encode`

  在python2的py文件里面写中文，则必须要添加一行声明文件编码的注释

  ```python
  # add on the first line
  # -*- coding:utf-8 -*-
  ```

#### 规范化注释

  标注好`Args`含义，以及输出`Returns`，并把输出类型与维度写出来。

  ```python
  # demo
  def build_label(x, pick=0,window=10):
      """
      build labels:   <----标注function的描述
      --------
      Args:           <----标注Args的描述
          x - x coordinate
          pick - the P or S arraivla times
          window - the bias of pick time
      Returns:        <----标注returns的描述
          labels: numpy.array [length, 1]
      """
      pick = int(pick)
      y = np.zeros(len(x))
      y[pick-window//2:pick+window//2] = np.ones(window)
      return y
  ```

- Vscode 内的批量注释、缩进：
  注释 `CTRL+K+C`
  取消注释 `CTRL+K+U`
  向左缩进 `CTRL+[`
  向右缩进 `CTRL+]`

___

## 4.Deep Learning

### 知识点

机器学习三步走：

1. 找 **Model**, 即 function set
  find a Model or a set of function.
2. **Goodness of Function**
  find the loss function.
    * Using input-output data to train.
    * Loss function L: $$L(f)=L(w,b)$$
      input: a function, output: how bad it is
    * pick the "Best" Function: aim to min Loss
    * Regularization:
      larger lambda, considering the training error less. We prefer smooth function, but don't be too smooth. Don't consider bias.
3. **Gradient Descent(Min Loss)**
  Gradient Descent 求导方法，来求最小值 in many iteration, 会陷入local optimal. But in linear regression, the loss function L is convex. So local optimal is the global optimal

**Overfitting**: A more complex model does not always lead to better performance on testing data.

**error come from**:

1. bias偏差
2. variance方差

Model more simpler, the function space is smaller, and the bias is larger/variance is smaller. Model is more complex, the function space is larger, and the bias is smaller/variance is larger.

So, **underfitting** solution:

- Add more features as input
- try A more complex model

**overfitting** solution:

- More data
- Regularization

**reduce the error of the testing set** : Using Cross Validation/N-fold Cross Validation.Try to divie the data in:

- training set
- validation set
- testing set

___

**————Gradient Descent————**

1. Learning Rate & GD
   - adaptive Learning Rate(Adagrad)
      definition: Divide the learning rates of each parameter by the root mean square of its previous derivatives.
      The best step is $\frac{|\partial f(x)|}{\partial ^2f(x)}$. The sigma is using first derivative to estimate second derivative.
   - Stochastic GD(SGD)
      definition: Loss is the summation over all training examples. However, SGD only pick an example to calculate the Loss and update the training parameters.
   - Feature Scaling(特征归一化)
      将不同数据缩放到相同尺度，减小不同数据对Loss的影响
2. Theory
    - Using the Taylor Series to express the Loss function. Solve the minimum by the inner product which means the GD.
    - learning rate need small enough
3. Limitation
   - Stuck at saddle point(拐点): 偏导为0
   - Stuck the local minima：偏导为0
   - vary slow at the plateau：偏导约为0

___

**-----Backpropagation-----**

由于求梯度的参数太多，要引入反向传播概念，其本质上还是GD，只是效率更高。
![](./figure/fig1.png)
链式求导法则：
![](./figure/fig2.png)
forward pass and backward pass
![](./figure/fig3.png)
式子中的C即为$Loss function$，z即为做完权重乘积和偏置相加后的值，但还未输入激活函数(activation function)里面.

- forward pass
  **计算$\frac{\partial z}{\partial w}$**
  ![](./figure/fig4.png)
  $\frac{\partial z}{\partial w}$的值即为上一层的输出值，此过程为forward pass

- backward pass
  **计算$\frac{\partial C}{\partial z}$**
  ![](./figure/fig5.png)
  式子$a = \sigma(z)$即为将z值输入激活函数后的输出值。假设网络构建就如图所示，那么$\frac{\partial z \prime}{\partial a} = w_3$，$\frac{\partial z \prime \prime}{\partial a} = w_4$，因此backward pass构建如下图所示：
  ![](./figure/fig6.png)
  图中的$\frac{\partial C}{\partial z \prime}$和$\frac{\partial C}{\partial z \prime \prime}$计算方法和$\frac{\partial C}{\partial z}$一样。因此整体的计算方法如下图所示：
  ![](./figure/fig7.png)

- summary
  ![](./figure/fig8.png)
  $a$即为上一层的输出，已知上一层的输出即可计算$\frac{\partial z}{\partial w}$，然后利用反向的计算$\frac{\partial C}{\partial z}$，二者乘积即可计算Loss对权重参数的求导：$\frac{\partial C}{\partial w}$

**在CNN里面，有多少个filter，输出就会是多少channels,他会自动考虑输入的channels，从而把filter变成立体的filter**

### 实例

1. 拾取地震到时
    - 建立工作文件夹：

___

## 5. Tensorflow/Keras

### 常用

#### load and save

  ```python
  from keras import models
  # ----- model save --------
  model.save(model_path)
  # ----- model load --------
  model = models.load_model(model_path)

  ```

#### 检查通道前后
  
  ```python
  from keras import backend as K
  # 判断训练要求通道放前还是放后面
  # channels 通道数为 3，lenght：数据长度
  if K.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 3, length)
      x_test = x_test.reshape(x_test.shape[0], 3, length)
      input_shape = (3, length) # 3 放在前面
  else:
      x_train = x_train.reshape(x_train.shape[0], length, 3)
      x_test = x_test.reshape(x_test.shape[0], length, 3)
      input_shape = (length, 3) # 3 放在后面
  ```

#### 模型可视化
  
  ```python
  from keras.utils.vis_utils import plot_model
  """
  plot model:
  ----------
  Args:
      model: model
      to_file: filename
      show_shapes: bool value, 是否显示shape
      dpi: resolution
  Returns:
      figure
  """
  plot_model(model, to_file='model.png', show_shapes=True, dpi=200)
  ```

### function

#### argmax

  ```python
  from keras import backend as K
  """
  using tensorflow as backend
  -----------
  argmax: 找出指定轴最大值的下标，'-1'即最后一个维度
  x: tensor or np.array
  for example: x 2*3维度
  x = [[1, 3, 5]
        6, 1, 8]]
  K.argmax(x, axis=-1)即为：
  output = [1, 0, 1]

  """
  K.argmax(x, axis=-1)
  ```

#### shape(-1)

  ```python
  """
  '-1': 未指定第一个维度值
  shape(-1, 3): 指定第二个维度值为3，第一个维度值会
  根据元素总数主动计算出来。
  """
  reshape(x, shape=(-1, 3))
  ```

#### **loss**

- sparse_categorical_crossentropy:

   ```python
   import keras
   keras.losses.sparse_categorical_crossentropy
   ```

   target 用 `one-hot` 编码，比如`[0, 1, 0]`就要使用 `categorical_crossentropy`
   target 没有用 `one-hot` 编码，而是直接用原始值，比如`[1, 2, 3]`，就要用`sparse_categorical_crossentropy`

- 正则化：
  正则项在优化过程中层的参数或层的激活值添加惩罚项，这些惩罚项将与损失函数一起作为网络的最终优化目标，惩罚项基于层进行惩罚，目前惩罚项的接口与层有关：`Dense, TimeDistributedDense, MaxoutDense, Covolution1D, Covolution2D, Convolution3D`具有共同的接口。
  - `kernel_regularizer`: 施加在**权重**上的正则项`keras.regularizers.Regularizer`
  - `bias_regularizer`: 施加在**偏置**向量上的正则项，为`keras.regularizer.Regularizer`
  - `activity_regularizer`：施加在**输出**上的正则项，为`keras.regularizer.Regularizer`
  for example:

  ```python
  from keras import regularizers

  model.add(Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.01)))
  ```

#### TFrecord Dataset

*environment： python2+tensorflow1*

- read the tfrecord data
    从`TFRecords`文件中读取数据， 首先需要用`tf.train.string_input_producer`生成一个解析队列。之后调用`tf.TFRecordReader的tf.parse_single_example`解析器。解析器首先读取解析队列，返回`serialized_example`对象，之后调用`tf.parse_single_example`操作将`Example`协议缓冲区(protocol buffer)解析为张量。
    for example:

    ```python
    #---------------------------------------------
    # case from the pick seismic data arrvail time
    #---------------------------------------------

    # preparation: Know what features in the data
    # -----
    # First: define the function of
    # tf.parse_single_example.
    #
    def _parse_example(serialized_example):
        """
        serialized_example: 解析器首先读取解析队列返回对象。
        -------
        return: singel data or labels. type: tensor
        -------
        n_traces: 3 channels
        win_size: data length
        """
        n_traces = 3
        win_size = 3001
        features = tf.parse_single_example(
            serialized_example,
            features={
                'window_size': tf.FixedLenFeature([], tf.int64),
                'n_traces': tf.FixedLenFeature([], tf.int64),
                'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'start_time': tf.FixedLenFeature([], tf.int64),
                'end_time': tf.FixedLenFeature([], tf.int64)})

        # Convert and reshape
        data = tf.decode_raw(features['data'], tf.float32)
        print ("data", data)
        data.set_shape([n_traces * win_size])
        data = tf.reshape(data, [n_traces, win_size])
        data = tf.transpose(data, [1, 0])
        # Pack
        features['data'] = data

        # Convert and reshape
        label = tf.decode_raw(features['label'], tf.float32)
        label.set_shape([n_traces * win_size])
        label = tf.reshape(label, [n_traces, win_size])
        label = tf.transpose(label, [1, 0])
        print ("****data,label", data, label.shape)
        # Pack
        features['label'] = label
        return features

    # -----
    # Second: read file and produce the file sequence
    filename = "XX.HSH1.tfrecords"
    pth = os.path.join("data","train",filename)

    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([pth, ], shuffle=True) # 读入流中
    _, serialized_example = reader.read(filename_queue) # 返回文件名和文件
    features = _parse_example(serialized_example)
    sample_inputt = features["data"]
    sample_target = features["label"] # 取出包含data和label的feature对象
    print(sample_inputt, sample_target) # print the shape

    # -----
    # Third: prodece session to change tensor to array
    #
    # some error: tensorflow CUDA out of memory 内存错误
    # there is the solution:
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    #开始不会给tensorflow全部gpu资源 而是按需增加
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)
        for i in range(200): # <---- read 200 data and label
            #在会话中取出data和label
            # 这里只读取了单个数据
            # 可以建立数组，将单个数据存储到数组中
            x_train, target = sess.run([sample_inputt,  sample_target])
            print(x_train.shape, target.shape)

    coord.request_stop()
    coord.join(thread)
    # -------
    # 上面代码读取的是单个的data和label
    # 若要读取 batch data, 要用tf.train.shuffle_batch读取
    # -------
    ```

- read data from tfrecord dataset(Change Tensor to Numpy Array):
  读取数据时出现卡住的情况，可以尝试以下解决方案

  ```python
  with tf.Session() as sess:
      """
      解释：
      --------
      coord = tf.train.Coordinator() 创建一个协调器，管理线程
      threads = tf.train.start_queue_runners(coord=coord) 启动QueueRunner,
      此时文件名队列已经进队
      """
      coord = tf.train.Coordinator()
      thread = tf.train.start_queue_runners(sess, coord)
      x_train, target = sess.run([x_train, target])
      print(x_train.shape, target.shape)# tensor have been changed to np.array
  ```

  `TensorFlow`提供了两个类来实现对Session中多线程的管理：`tf.Coordinator`和 `tf.QueueRunner`，这两个类往往一起使用。
  - `Coordinator`类用来管理在`Session`中的多个线程，可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常，该线程捕获到这个异常之后就会终止所有线程。使用 `tf.train.Coordinator()`来创建一个线程管理器（协调器）对象。
  - `QueueRunner`类用来启动`tensor`的入队线程，可以用来启动多个工作线程同时将多个`tensor`（训练数据）推送入文件名称队列中，具体执行函数是 `tf.train.start_queue_runners`。
    只有调用 `tf.train.start_queue_runners` 之后，才会真正把`tensor`推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态。

- total sample number in tfrecords data

  ```python
  # 该函数用于统计 TFRecord 文件中的样本数量(总数)
  def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return  sample_nums
  ```

#### `ERROR`

  程序过程中遇到的错误及其solution

- OOM:

  ```cmd
  TensorFlow: Resource exhausted: OOM when allocating tensor with shape[256, 512， 16, 16]
  ```

  `[256, 512， 16, 16]`的第一个参数表示`batch_size`的大小，第二个参数表示某层卷积核的个数，第三个参数表示图像的高，第四个参数表示图像的长 这里出现这种错误的原因时超出内存了，因此可以适当减小`batch_size`的大小即可解决,或者将卷积核变小，减少参数。

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>