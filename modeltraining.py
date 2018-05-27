import bokeh
import csv
import datetime
import itertools
import keras
import math
import matplotlib.pyplot as plt
# LSTM for international airline passengers problem with window regression framing
import numpy
import numpy as np
import pandas as pd
import pickle
import requests
import time
import urllib
from bokeh.plotting import figure, show
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# keys = ["USDT_BTC", "BTC_LTC" ,"BTC_NXT", "BTC_DOGE", "BTC_ETH", "BTC_BCH"]
keys = ["BTC_LTC", "BTC_NXT", "BTC_DOGE", "BTC_ETH", "BTC_BCH"]


def read_data(filename):
    with open(filename, mode="r") as infile:
        reader = csv.reader(infile)

        data = {}
        headers = []
        for row in reader:
            if not headers:
                headers = list(row)
            else:
                for cur_header, value in zip(headers, row):
                    if cur_header not in data.keys():
                        data[cur_header] = []
                    if cur_header == "Date Hour":
                        date, hour = value.split(" ")
                        year, month, day = date.split("-")
                        hh, mm, ss = hour.split(":")
                        time = datetime.datetime(year=int(year),
                                                 month=int(month),
                                                 day=int(day),
                                                 hour=int(hh), minute=int(mm),
                                                 second=int(ss))

                        value = time.timestamp()
                    data[cur_header].append(float(value))

        return data


from ratelimiter import RateLimiter


def get_poloniex_data(url, payload):
    return requests.get(url, params=payload)


def get_data(in_size):
    url = "https://poloniex.com/public"

    currency_pairs = keys
    start_stamp = 1410158341
    end_stamp = time.time()
    chunk_size = 600000
    start_stamps = np.arange(start_stamp, end_stamp - chunk_size, chunk_size)
    end_stamps = np.arange(start_stamp + chunk_size, end_stamp, chunk_size)

    data_frames = {}
    ratelimiter = RateLimiter(max_calls=1, period=1 / 5)
    for pair in currency_pairs:
        frame = None
        all_data_lists = []
        for start, end in zip(start_stamps, end_stamps):
            start_chunk = start
            end_chunk = end
            chunk_chunk = chunk_size

            while True:
                payload = {"command": "returnTradeHistory", "currencyPair": pair,
                           "start": start_chunk, "end": end_chunk}

                with ratelimiter:
                    req = get_poloniex_data(url, payload)

                data_list_dict = req.json()

                if len(data_list_dict) < 50000 and end_chunk == end:
                    all_data_lists += data_list_dict
                    print("Got all the items")
                    break
                elif len(data_list_dict) < 50000:
                    print("Appending {} items to list".format(len(data_list_dict)))
                    all_data_lists += data_list_dict
                    start_chunk = end_chunk
                    end_chunk = start_chunk + chunk_chunk
                    if end_chunk > end:
                        end_chunk = end
                else:

                    chunk_chunk = int(chunk_chunk / 2)
                    start_chunk = start_chunk
                    end_chunk = start_chunk + chunk_chunk

                    print("Redoing the call.., chunk size is now: {}".format(chunk_chunk))
                    if end_chunk > end:
                        end_chunk = end

            if all_data_lists:
                print("{} now has {} items".format(pair,
                                                   len(all_data_lists)))

            print("For {} got {} items".format(pair, len(data_list_dict)))
        frame = pd.DataFrame(all_data_lists)

        print("Got data frame with {} items".format(len(frame)))
        data_frames[pair] = frame

        # with open("train_data.pickle", "wb") as f:
        #    pickle.dump(data_frames, f)


store_processed = pd.HDFStore('processed_data.h5')

max_integer = 1
zero_integer = 0
min_integer = -max_integer
rate_in_max_level = 1
rate_out_max_level = 1

step_s = 3
in_len_h = 48
in_size = int((in_len_h * 60 * 60) / step_s)

out_mins_resolution = 60

down_sample_factor = 2000
downsample_items = int(in_size / down_sample_factor)

downsample_step = str(out_mins_resolution) + "T"
out_size = int((out_mins_resolution * downsample_items * 60) / step_s)

nrows_buy = int(store_processed.get_storer("buy").nrows - out_size)
nrows_sell = int(store_processed.get_storer("sell").nrows - out_size)
chunks_read = int(in_size + out_size)


def generate_io(chunk_data_buy_in, chunk_data_sell_in,
                plot=False,
                keys=keys,
                min_integer=min_integer,
                out_mins_resolution=15):
    downsample_step = str(out_mins_resolution) + "T"
    artificial_scale = 300
    scaler = artificial_scale * out_mins_resolution * 60

    if plot:
        p1 = bokeh.plotting.figure(title="Rate ", x_axis_type="datetime")
        p2 = bokeh.plotting.figure(title="Amount ", x_axis_type="datetime")

    chunk_data_buy = chunk_data_buy_in.resample(downsample_step).mean()
    chunk_data_sell = chunk_data_sell_in.resample(downsample_step).mean()

    in_size = len(chunk_data_buy)

    num_vars_per_key = 5
    output_rows = num_vars_per_key * len(keys)
    in_data = np.zeros((output_rows, in_size))
    for key, rate_row_buy, rate_row_sell, amount_row_buy, amount_row_sell, time_row, result_row in zip(keys,
                                                                                                       range(0,
                                                                                                             output_rows,
                                                                                                             num_vars_per_key),
                                                                                                       range(1,
                                                                                                             output_rows,
                                                                                                             num_vars_per_key),
                                                                                                       range(2,
                                                                                                             output_rows,
                                                                                                             num_vars_per_key),
                                                                                                       range(3,
                                                                                                             output_rows,
                                                                                                             num_vars_per_key),
                                                                                                       range(4,
                                                                                                             output_rows,
                                                                                                             num_vars_per_key),
                                                                                                       range(0, len(
                                                                                                           keys))):
        # print("Converting {}".format(key))
        rate_label = "{}_rate".format(key)
        amount_label = "{}_amount".format(key)
        in_data[rate_row_buy, :] = ((chunk_data_buy[rate_label][
                                     :].as_matrix() / rate_in_max_level) * max_integer * scaler) + zero_integer
        in_data[rate_row_sell, :] = ((chunk_data_sell[rate_label][
                                      :].as_matrix() / rate_in_max_level) * max_integer * scaler) + zero_integer
        in_data[amount_row_buy, :] = (chunk_data_buy[amount_label][:].as_matrix() * max_integer)
        in_data[amount_row_sell, :] = (chunk_data_sell[amount_label][:].as_matrix() * max_integer)

        hours = chunk_data_sell.index[0:in_size].hour * 3600
        mins = chunk_data_sell.index[0:in_size].minute * 60
        secs = chunk_data_sell.index[0:in_size].second
        time_of_day = (hours + mins + secs) / (60 * 60 * 24)
        in_data[time_row, :] = time_of_day * max_integer

        in_data[in_data > max_integer] = max_integer
        in_data[in_data < min_integer] = min_integer

        if plot:
            pal = bokeh.palettes.Category20c[20] + bokeh.palettes.Category20c[20]
            p1.line(chunk_data_buy.index[0:in_size], in_data[rate_row_buy, :],
                    color=pal[rate_row_buy], legend="buy" + key)

            p1.line(chunk_data_sell[0:in_size].index, in_data[rate_row_sell, :], color=pal[rate_row_sell],
                    legend="sell " + key)

            p2.line(chunk_data_buy[0:in_size].index, time_of_day, color=pal[rate_row_buy], legend="buy " + key)
            # p2.line(chunk_data_buy[0:in_size].index, in_data[amount_row_buy, :], color=pal[rate_row_buy], legend="buy " + key)
            # p2.line(chunk_data_sell[0:in_size].index, in_data[amount_row_sell, :], color=pal[rate_row_buy], legend="sell " + key)

    if plot:
        bokeh.plotting.show(p1)
        bokeh.plotting.show(p2)
    in_data = in_data.conj().transpose()
    in_data = np.reshape(in_data, (1, in_data.shape[0], in_data.shape[1]))

    return in_data


def data_generator(file_name='processed_data.h5', seed=1):
    keys = ["BTC_LTC", "BTC_NXT", "BTC_DOGE", "BTC_ETH", "BTC_BCH"]
    store_processed = pd.HDFStore(file_name)

    import random
    import math

    step_s = 3
    in_len_h = 62
    in_size = int((in_len_h * 60 * 60) / step_s)

    in_mins_resolution = 15
    elems_tot = math.floor(in_size / ((in_mins_resolution * 60) / step_s))

    nrows_buy = int(store_processed.get_storer("buy").nrows - in_size + 2)
    nrows_sell = int(store_processed.get_storer("sell").nrows - in_size + 2)
    chunks_read = int(in_size)

    nrows_buy = int(store_processed.get_storer("buy").nrows - in_size + 2)
    nrows_sell = int(store_processed.get_storer("sell").nrows - in_size + 2)
    chunks_read = int(in_size)
    random.seed(seed)
    while True:
        # for i, j in zip(range(nrows_buy//chunks_read + 1), range(nrows_sell//chunks_read + 1)):

        i = random.randint(0, nrows_buy - chunks_read + 1)
        j = i
        chunk_buy = store_processed.select('buy',
                                           start=i,
                                           stop=(i + chunks_read))
        chunk_sell = store_processed.select('sell',
                                            start=i,
                                            stop=(i + chunks_read))

        plot = False
        if i % 50000:
            plot = False

        in_arr = generate_io(chunk_buy, chunk_sell, plot, keys=keys)
        out_arr = in_arr[:, -1, :]
        in_arr_crop = in_arr[:, 0:-1, :]
        if in_arr_crop.shape[1] == elems_tot:
            yield in_arr_crop, out_arr
            # else:
            #    print("in arr shape: {}, expected elements: {}".format(in_arr.shape, elems_tot))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
import time

in_len_h = 62
in_size = int((in_len_h * 60 * 60) / step_s)

chunks_read = int(in_size)

i = 230
chunk_buy = store_processed.select('buy',
                                   start=i * chunks_read,
                                   stop=(i + 1) * chunks_read)
chunk_sell = store_processed.select('sell',
                                    start=i * chunks_read,
                                    stop=(i + 1) * chunks_read)

train_x = generate_io(chunk_buy, chunk_sell, plot=False)
train_y = train_x[:, -1, :]
train_x = train_x[:, 0:-1, :]

i = 200
chunk_buy = store_processed.select('buy',
                                   start=i * chunks_read,
                                   stop=(i + 1) * chunks_read)
chunk_sell = store_processed.select('sell',
                                    start=i * chunks_read,
                                    stop=(i + 1) * chunks_read)

test_x = generate_io(chunk_buy, chunk_sell)
test_y = test_x[:, -1, :]
test_x = test_x[:, 0:-1, :]

from keras import backend as K


def mean_absolute_error_non_sym(y_true, y_pred):
    diff = K.mean(K.abs(y_pred - y_true), axis=-1)
    if K.sign(K.mean(y_pred, axis=-1)) is not K.sign(K.mean(y_true, axis=-1)):
        diff = diff * 10

    return diff


def valley_loss(y_true, y_pred):
    diff = K.abs(((100. * (y_true - y_pred)) / K.clip(K.abs(y_pred),
                                                      0.4,
                                                      None)))
    return K.mean(diff, axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    """

    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)

    return K.mean(_logcosh(y_pred - y_true), axis=-1)


class SGDLearningRateTracker(object):
    def __init__(args):
        model = None

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

    def set_model(self, model=None):
        self.model = model

    def set_params(self, params=None):
        self.params = params


# nb_filters=32
kernel_size = (3, 3)

from keras import backend as K

K.clear_session()
# K.set_learning_phase(0)

from datetime import datetime

now = datetime.now()

try:
    # del model
    # del tb_call
    pass
except:
    print("All good!")

loss = mean_absolute_error_non_sym

tb_call = keras.callbacks.TensorBoard(
    log_dir='/home/archScifi/tmp/board/{}'.format(now.strftime(str(loss) + "_%Y%m%d-%H%M%S")), histogram_freq=1,
    write_graph=True, write_images=True)

checkpointer = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=False)

epochs = 200
initial_lr = 1e-2
momentum = 0.3
decay = initial_lr / epochs

# nb_filters=32
kernel_size = (3, 3)

from keras import backend as K

K.clear_session()
# K.set_learning_phase(0)

from datetime import datetime

now = datetime.now()

try:
    # del model
    # del tb_call
    pass
except:
    print("All good!")

loss = mean_absolute_error_non_sym

tb_call = keras.callbacks.TensorBoard(
    log_dir='/home/archScifi/tmp/board/{}'.format(now.strftime(str(loss) + "_%Y%m%d-%H%M%S")), histogram_freq=1,
    write_graph=True, write_images=True)

checkpointer = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=False)

epochs = 200
initial_lr = 1e-2
momentum = 0.3
decay = initial_lr / epochs

model.fit(train_x, train_y, epochs=10, batch_size=1, verbose=2, callbacks=[tb_call, reduce_lr],
          validation_data=(test_x, test_y))

K.set_value(model.optimizer.lr, initial_lr * 1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                                              patience=10, min_lr=1e-5, verbose=1, cooldown=10)

print("Setting decay at {}".format(decay))

gen_obj = data_generator(seed=1)
print(K.get_value(model.optimizer.lr))
model.fit_generator(gen_obj, epochs=900, samples_per_epoch=128, verbose=1, callbacks=[tb_call, checkpointer, reduce_lr],
                    use_multiprocessing=True, validation_data=(test_x, test_y))

# validation_data=(test_x, test_y)


i = 200
chunk_buy_v = store_processed.select('buy',
                                     start=i * chunks_read,
                                     stop=(i + 1) * chunks_read + chunks_read)
chunk_sell_v = store_processed.select('sell',
                                      start=i * chunks_read,
                                      stop=(i + 1) * chunks_read + chunks_read)

ver_x_all = generate_io(chunk_buy_v, chunk_sell_v, plot=False)
ver_x = ver_x_all[:, 0:input_len, :]
real_y = ver_x_all[:, input_len:input_len * 2, :]

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
