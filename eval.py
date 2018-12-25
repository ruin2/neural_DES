import codecs
import tensorflow as tf
import numpy as np
import json
import re
from collections import Counter

#V3版本已经能够猜对5个左右的字符，猜测，可能是数据量不足，或者是参数空间不足够拟合DES的全部变换
#V4版本采用全新的数据集，共有13.4M的英文小说做成数据去拟合DES的全部映射参数
#V5版本是更换了数据集的制作方式，每4个英文字母加密一次，因为DES的分组密码特性，64位比特相当于两个英文字母
#V7版本的代码内容和V5一样，只是采用了师兄的建议，首先分析5000对明密文攻击的可行性，
#V8版本加了一个统计数据,统计错误出现的位置，发现了强相关性,可以用来降低破解的时间
#TODO 加入summary
#TODO 加入之前论文10的反向分析
#TODO 加入tfargs 控制可变参数
#TODO 计算错误共现概率



#比较好的实验结果如下：3层x100,5x60 参数量为300
#餐数量200的效果也不错

#加入个统计方法
error_distribution = Counter()



def fetch_error_distribution(data):#用于查看错误的位置
    #然后又加入了一些改动，把这些分布写到一个文件中去。
    fr = codecs.open('error_distribution','w',encoding = 'utf-8')
    mylist = []
    for index,item in enumerate(data):
        if not item:
            mylist.append(index)
            error_distribution[index] += 1
    fr.write(json.dumps(mylist)+'\n')
    return error_distribution,mylist

def clean(data):
    # 用于将队列的返回结果进行重新构建
    data = np.array(json.loads(data))
    return np.reshape(data, [-1, 1])

def func(x):
    return tf.cast(tf.greater(x,0.5),tf.float32)

train_data = []
target_data = []
#我在V8版本采用了最大的数据集，这样做的目的就是可以不需要验证集
with codecs.open(r'C:\Users\ruin2\DES密码分析\train_data/bytes_data_1203','r',encoding = 'utf-8') as fr:
    mydata = fr.readlines()
    for line in mydata:
        line = line.strip('\n')
        line = line.split('###')
        #左边是明文，右边是密文
        #明文是label，密文是data
        target_data.append(line[0])
        train_data.append(line[1])
fr.close()
del mydata


# 开始搭建网络
def fc(inputs, units, layer_name,activation = None):  # 只在最后一层添加relu作为激活
    output = tf.layers.dense(inputs=inputs,
                             units=units,
                             activation=activation,
                             name=layer_name)
    return output

def get_weight(shape, name):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)  # 服从均匀正太分布
    return tf.Variable(initial, name=name)

def get_bias(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def inference_loop(inputs):  # 将【72，】的密文作为模型的输入
    # num_neural = 10#每一层的神经元数量  #ada 的时候，30以内会取得不错的效果,但奇怪的是，使用SGD就会出现NAN。
    # ori = inputs
    num_neural = 30
    for i in range(2):
        name = str(i)
        # if i % 3 == 0:#skip layer#在V5版本中，加上skip层的效果明显不好
        #     inputs = fc(inputs, num_neural, name) + ori
        #     continue
        inputs = fc(inputs, num_neural ,layer_name = name)#输出的大小是
    final_w_l = get_weight([64, 64], 'final_w_l')#左乘矩阵

    final_w_r = get_weight([num_neural,1],'final_w_r')#右乘矩阵
    logits = tf.matmul(final_w_l,inputs)
    logits = tf.matmul(logits,final_w_r)
    return logits

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        # logits = tf.cast(logits,tf.int32)#这个表达不是很好，所以我们在V5版本换成tf.map_fn,这样的话，就可以用0.5作为分割界限了
        logits = tf.map_fn(lambda x : func(x),logits)
        logits = tf.cast(logits,tf.int32)
        labels = tf.cast(labels,tf.int32)
        correct = tf.equal(logits,labels)
        correct_list = correct
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_sum(correct)
        # tf.summary.scalar('accuracy',accuracy)
    return accuracy,correct_list


def losses(data, labels):
    with tf.name_scope('losses') as scope:
        loss = tf.losses.absolute_difference(labels=labels,predictions=data)
        # tf.summary.scalar('loss',loss)
    return loss


def trainning(loss):
    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss)
    return train_op


xs = tf.placeholder(tf.float32, shape=[64, 1])
ys = tf.placeholder(tf.float32, shape=[64, 1])

train_logits = inference_loop(xs)
train_loss = losses(train_logits, ys)
summary_loss = tf.summary.scalar('loss',train_loss)
train_step = trainning(train_loss)
train_acc = evaluation(train_logits,ys)
accuracy = tf.cast(train_acc[0],tf.int32)
summary_acc = tf.summary.scalar('accuracy',accuracy)
summary_op = tf.summary.merge([summary_loss,summary_acc])
# sum_ops = tf.summary.merge_all()

queue = tf.FIFOQueue(capacity=2000, dtypes=[tf.string, tf.string], )
enqueue_op = queue.enqueue_many([train_data, target_data])
x, y = queue.dequeue()
qr = tf.train.QueueRunner(queue, [enqueue_op])

# with tf.Graph().as_default():
#     session_conf = tf.ConfigProto(
#         allow_soft_placement=True,
#         log_device_placement=False)
#     sess = tf.Session(config=session_conf)
#     with sess.as_default():
with tf.Session() as sess:
    sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
    checkpoint_dir = './model'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    train_summary_writer = tf.summary.FileWriter('evals/',sess.graph)

    for i in range(2000):
        datas, labels = sess.run([x, y])
        loss, acc,summaries = sess.run([train_loss, train_acc,summary_op],
                                           feed_dict={xs: clean(datas),
                                                      ys: clean(labels)})
        train_summary_writer.add_summary(summaries, i)
        fetch_error_distribution(acc[1])
        if i % 100 == 0:
            print('step = {},loss = {:.2f},acc = {}'.format(i,loss,acc[0]))




    coord.request_stop()
    # coord.join(threads)
    sess.close()

    #tensorboard --logdir C:\Users\ruin2\PycharmProjects\untitled\logs
    #启动tensorboard的时候，参数不要用带''的这种。当成linux操作



