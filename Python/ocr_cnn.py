import os
import cv2
import numpy as np
import tensorflow as tf


LABEL_NUM = 28
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
dict_list = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
             9: "a", 10: "b", 11: "c", 12: "d", 13: "e", 14: "f", 15: "g", 16: "h",
             17: "K", 18: "Q", 19: "R", 20: "B", 21: "N",
             22: "x", 23: "=", 24: "-", 25: "+", 26: "#",
             27: " "}


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, LABEL_NUM])
keep_prob = tf.placeholder(tf.float32)  # dropout


def test(_para):
    return _para


def resize_file(_folder, _folder2, _folder3):
    _i = 0
    _j = 0
    _folders = os.listdir(_folder)
    for _index in range(LABEL_NUM):
        _folder_sub = _folder + "/" + _folders[_index] + "/"
        _files = os.listdir(_folder_sub)
        for _file in _files:
            if _file[-4:] != '.png':
                continue
            _image = cv2.imdecode(np.fromfile(_folder_sub + _file, dtype=np.uint8), 0)
            _row, _col = _image.shape
            if np.max(_image)-np.min(_image) >= 45:
                _, _image = cv2.threshold(_image, 0, 255, cv2.THRESH_OTSU)
                _image = ~_image
                if _row * IMAGE_WIDTH > _col * IMAGE_HEIGHT:
                    _col2 = int(IMAGE_WIDTH * _row / IMAGE_HEIGHT)
                    _image2 = np.zeros((_row, _col2), dtype=np.uint8)
                    _image2[:, int((_col2-_col)/2):int((_col2-_col)/2)+_col] = _image[:, :]
                    _image = cv2.resize(_image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
                elif _row * IMAGE_WIDTH < _col * IMAGE_HEIGHT:
                    _row2 = int(IMAGE_HEIGHT * _col / IMAGE_WIDTH)
                    _image2 = np.zeros((_row2, _col), dtype=np.uint8)
                    _image2[int((_row2-_row)/2):int((_row2-_row)/2)+_row, :] = _image[:, :]
                    _image = cv2.resize(_image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
                else:
                    _image = cv2.resize(_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                cv2.imwrite(_folder2 + "/" + _folders[_index] + "/" + _file, _image)
                _i += 1
            else:
                # _image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                cv2.imwrite(_folder3 + "/" + _folders[_index] + "/" + _file, _image)
                _j += 1
    return _i, _j


def label_file(_folder):
    _i = 0
    _folders = os.listdir(_folder)
    _file = open(_folder + "/data_list.txt", 'w')
    _file2 = open(_folder + "/data_list2.txt", 'w')
    for _index in range(LABEL_NUM):
        _folder_sub = _folder + "/" + _folders[_index] + "/"
        _images = os.listdir(_folder_sub)
        _label = _folders[_index].split('_')[0]
        for _image in _images:
            if _image[-4:] != '.png':
                continue
            if _i % 10 < 7:
                _file.write(_folder_sub + _image + "|" + _label + "\n")
            else:
                _file2.write(_folder_sub + _image + "|" + _label + "\n")
            _i += 1
    _file.close()
    _file2.close()
    return _i


def encode_to_tfrecode(_file_p, _data_root, _new_name='data.tfrecodes', 
                       _resize=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    _count = 0
    _writer = tf.python_io.TFRecordWriter(_data_root + '/' + _new_name)  # 生成的data文件的文件名，文件路径
    with open(_file_p, 'r') as _file:
        for _line in _file.readlines():  # 遍历每张图片以及标签
            _words = _line.split('|')
            _image = cv2.imdecode(np.fromfile(_words[0], dtype=np.uint8), 0)
            if _resize is not None:
                _image = cv2.resize(_image, _resize)
            _height, _width = _image.shape
            _label = int(_words[1])
            _example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[_height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[_width])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[_image.tobytes()])),  # 修改特征
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[_label]))
            }))
            _serialized = _example.SerializeToString()
            _writer.write(_serialized)
            _count += 1
    _writer.close()
    print(label_file, "sample count:", _count)
    return _count


def decode_from_tfrecode(_filename, _num_epoch=None):
    _filename_queue = tf.train.string_input_producer([_filename], num_epochs=_num_epoch)
    _reader = tf.TFRecordReader()
    _, _serialized = _reader.read(_filename_queue)
    _example = tf.parse_single_example(_serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    _label = tf.cast(_example['label'], tf.int32)
    _image = tf.decode_raw(_example['image'], tf.uint8)
    _image = tf.reshape(_image, tf.stack([
        tf.cast(_example['height'], tf.int32),
        tf.cast(_example['width'], tf.int32),
    ]))
    return _image, _label


def get_batch(_image, _label, _batch_size=100):
    _distorted_image = tf.reshape(_image, [IMAGE_HEIGHT * IMAGE_WIDTH])
    _image, _label_batch = tf.train.shuffle_batch([_distorted_image, _label], batch_size=_batch_size, 
                                                  num_threads=16,
                                                  capacity=50000, min_after_dequeue=1000)
    return _image, tf.reshape(_label_batch, [_batch_size, 1])


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([4 * 4 * 64, 140]))
    b_d = tf.Variable(b_alpha * tf.random_normal([140]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([140, LABEL_NUM]))
    b_out = tf.Variable(b_alpha * tf.random_normal([LABEL_NUM]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def one_hot(_label):
    _num = _label.shape[0]
    _label_r = np.zeros([_num, LABEL_NUM])
    for _i in range(_num):
        _label_r[_i][_label[_i]] = 1
    return _label_r


def train_crack_captcha_cnn(_label_file, _test_label_file, _model_path, _old_model="", 
                            min_acc=0.99, max_repeat=3000):
    if min_acc < 0.8:
        min_acc = 0.8
    if max_repeat < 99:
        max_repeat = 99
    _output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    _predict = tf.reshape(_output, [-1, LABEL_NUM])
    max_idx_p = tf.argmax(_predict, 1)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, LABEL_NUM]), 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # train data
    image, label = decode_from_tfrecode(_label_file)
    batch_x, batch_y = get_batch(image, label, _batch_size=100)
    # test data
    test_image, test_label = decode_from_tfrecode(_test_label_file)
    test_batch_x, test_batch_y = get_batch(test_image, test_label, _batch_size=100)
    # init
    _saver = tf.train.Saver()
    _init = tf.global_variables_initializer()
    _config = tf.ConfigProto()
    _config.log_device_placement = True
    _config.gpu_options.allow_growth = True
    # train
    with tf.Session(config=_config) as _sess:
        _sess.run(_init)
        if _old_model != "":
            try:
                _saver.restore(_sess, _old_model + "/crack_capcha.model")
                print("old model load succeed!")
            except:
                print("error: old model not exist!")
        step = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while True:
            image_np, label_np = _sess.run([image, label])
            batch_x_r, batch_y_r = _sess.run([batch_x, batch_y])
            batch_y_r = one_hot(batch_y_r)
            _, loss_ = _sess.run([optimizer, loss], feed_dict={X: batch_x_r, Y: batch_y_r, keep_prob: 0.75})
            print(step, loss_)
            if step % 50 == 0:
                image_np, label_np = _sess.run([test_image, test_label])
                batch_x_r, batch_y_r = _sess.run([test_batch_x, test_batch_y])
                batch_y_r = one_hot(batch_y_r)
                acc = _sess.run(accuracy, feed_dict={X: batch_x_r, Y: batch_y_r, keep_prob: 1.})
                print('accuracy:', step, acc)
                # if (step > max_repeat) or ((_old_model != "") and (step > max_repeat/10)):
                if (acc > min_acc) or (step > max_repeat):  # 0.95
                    _saver.save(_sess, _model_path + "/crack_capcha.model")
                    break
            step += 1
        coord.request_stop()
        coord.join(threads)
    return acc


print("load ocr_cnn.pyc succeed!")
# FOLDER = "E:/共享文件夹/_TEMP/国际象棋/dataset"
FOLDER = "./data/src"
FOLDER2 = "./data/dataset"
FOLDER3 = "./data/badset"
FOLDER4 = "./data/model"
# print(resize_file(FOLDER, FOLDER2, FOLDER3))
# print('total:', label_file(FOLDER2))
# print('train sample count:', encode_to_tfrecode(FOLDER2 + "/data_list.txt", FOLDER2, _new_name='traindata.tfrecodes'))
# print('test sample count:', encode_to_tfrecode(FOLDER2 + "/data_list2.txt", FOLDER2, _new_name='testdata.tfrecodes'))
train_crack_captcha_cnn(FOLDER2 + "/traindata.tfrecodes", FOLDER2 + "/testdata.tfrecodes", FOLDER4,
                        _old_model="", min_acc=0.98, max_repeat=1000)

# label_file(FOLDER + "train")
# encode_to_tfrecode(FOLDER + "train/data_list.txt", FOLDER, _new_name='traindata.tfrecodes')
# encode_to_tfrecode(FOLDER + "train/data_list2.txt", FOLDER, _new_name='testdata.tfrecodes')
# train_crack_captcha_cnn(FOLDER + "traindata.tfrecodes", FOLDER + "testdata.tfrecodes", FOLDER + "model",
#                         _old_model="", min_acc=0.98, max_repeat=600)
