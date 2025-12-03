# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import argparse
import json
import time
import datetime
import numpy as np
import tensorflow as tf
from config import *
from get_data import get_current_number, spider
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", default="ssq", type=str, help="选择训练数据: 双色球/大乐透"
)
parser.add_argument("--num", default=10, type=int, help="生成预测号码的数量")
args = parser.parse_args()

# 关闭eager模式
tf.compat.v1.disable_eager_execution()


def load_model(name):
    """加载模型"""
    if name == "ssq":
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(
                    model_args[args.name]["path"]["red"]
                )
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(
            red_sess,
            "{}red_ball_model.ckpt".format(model_args[args.name]["path"]["red"]),
        )
        logger.info("已加载红球模型！")

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(
                    model_args[args.name]["path"]["blue"]
                )
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(
            blue_sess,
            "{}blue_ball_model.ckpt".format(model_args[args.name]["path"]["blue"]),
        )
        logger.info("已加载蓝球模型！")

        # 加载关键节点名
        with open("{}/{}/{}".format(model_path, args.name, pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(args.name)
        logger.info(
            "【{}】最近一期:{}".format(name_path[args.name]["name"], current_number)
        )
        return red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number
    else:
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(
                    model_args[args.name]["path"]["red"]
                )
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(
            red_sess,
            "{}red_ball_model.ckpt".format(model_args[args.name]["path"]["red"]),
        )
        logger.info("已加载红球模型！")

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(
                    model_args[args.name]["path"]["blue"]
                )
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(
            blue_sess,
            "{}blue_ball_model.ckpt".format(model_args[args.name]["path"]["blue"]),
        )
        logger.info("已加载蓝球模型！")

        # 加载关键节点名
        with open("{}/{}/{}".format(model_path, args.name, pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(args.name)
        logger.info(
            "【{}】最近一期:{}".format(name_path[args.name]["name"], current_number)
        )
        return red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number


def get_year():
    """截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(mode, name, predict_features, windows_size):
    """处理异常"""
    if mode:
        return predict_features
    else:
        if len(predict_features) != windows_size:
            logger.warning(
                "期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！"
            )
            last_current_year = (get_year() - 1) * 1000
            max_times = 160
            while len(predict_features) != 3:
                predict_features = spider(
                    name,
                    last_current_year + max_times,
                    get_current_number(name),
                    "predict",
                )[[x[0] for x in ball_name]]
                time.sleep(np.random.random(1).tolist()[0])
                max_times -= 1
            return predict_features
        return predict_features


# 修改 get_red_ball_predict_result 函数
def get_red_ball_predict_result(
    red_graph, red_sess, pred_key_d, predict_features, sequence_len, windows_size, n
):
    """获取多个红球预测结果"""
    name_list = [(ball_name[0], i + 1) for i in range(sequence_len)]
    base_data = (
        predict_features[
            ["{}_{}".format(name[0], i) for name, i in name_list]
        ].values.astype(int)
        - 1
    )

    preds = []
    # 改为循环单样本预测
    for _ in range(n):
        # 可添加微小随机扰动增加多样性
        data = base_data + np.random.randint(-1, 2, size=base_data.shape) * 0.1
        data = data.clip(min=0).astype(int)

        with red_graph.as_default():
            reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(
                pred_key_d[ball_name[0][0]]
            )
            pred = red_sess.run(
                reverse_sequence,
                feed_dict={
                    "inputs:0": data.reshape(
                        1, windows_size, sequence_len
                    ),  # 保持(1, 100, 6)形状
                    "sequence_length:0": np.array([sequence_len]),
                },
            )
        preds.append(pred[0])  # 取第一个维度的结果

    return np.array(preds), name_list


# 同理修改 get_blue_ball_predict_result 函数中预测部分
def get_blue_ball_predict_result(
    blue_graph,
    blue_sess,
    pred_key_d,
    name,
    predict_features,
    sequence_len,
    windows_size,
    n,
):
    """获取多个蓝球预测结果"""
    if name == "ssq":
        base_data = predict_features[[ball_name[1][0]]].values.astype(int) - 1
        preds = []
        for _ in range(n):
            data = base_data + np.random.randint(-1, 2, size=base_data.shape) * 0.1
            data = data.clip(min=0).astype(int)

            with blue_graph.as_default():
                softmax = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    pred_key_d[ball_name[1][0]]
                )
                pred = blue_sess.run(
                    softmax,
                    feed_dict={
                        "inputs:0": data.reshape(1, windows_size)
                    },  # 保持单样本形状
                )
            preds.append(pred[0])
        return np.array(preds)
    else:
        # 大乐透蓝球部分也做类似修改，保持单样本输入
        name_list = [(ball_name[1], i + 1) for i in range(sequence_len)]
        base_data = (
            predict_features[
                ["{}_{}".format(name[0], i) for name, i in name_list]
            ].values.astype(int)
            - 1
        )

        preds = []
        for _ in range(n):
            data = base_data + np.random.randint(-1, 2, size=base_data.shape) * 0.1
            data = data.clip(min=0).astype(int)

            with blue_graph.as_default():
                reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    pred_key_d[ball_name[1][0]]
                )
                pred = blue_sess.run(
                    reverse_sequence,
                    feed_dict={
                        "inputs:0": data.reshape(1, windows_size, sequence_len),
                        "sequence_length:0": np.array([sequence_len]),
                    },
                )
            preds.append(pred[0])
        return np.array(preds), name_list


def get_final_result(
    red_graph,
    red_sess,
    blue_graph,
    blue_sess,
    pred_key_d,
    name,
    predict_features,
    mode=0,
    n=10,  # 传递批量预测的样本数量
):
    """最终预测函数，返回多个预测结果"""
    m_args = model_args[name]["model_args"]
    if name == "ssq":
        red_preds, red_name_list = get_red_ball_predict_result(
            red_graph,
            red_sess,
            pred_key_d,
            predict_features,
            m_args["sequence_len"],
            m_args["windows_size"],
            n,  # 获取 n 个红球预测结果
        )

        blue_preds = get_blue_ball_predict_result(
            blue_graph,
            blue_sess,
            pred_key_d,
            name,
            predict_features,
            0,
            m_args["windows_size"],
            n,  # 获取 n 个蓝球预测结果
        )

        ball_name_list = [
            "{}_{}".format(name[mode], i) for name, i in red_name_list
        ] + [ball_name[1][mode]]

        # 生成并去重结果
        pred_result_list = []
        seen = set()
        for i in range(n):
            # 将蓝球整数转为列表（如 12 → [12]）
            pred_result = red_preds[i].tolist() + [blue_preds[i].item()]
            result_dict = {
                b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result)
            }
            # 转换为可哈希的元组用于去重
            result_tuple = tuple(sorted(result_dict.items()))
            if result_tuple not in seen:
                seen.add(result_tuple)
                pred_result_list.append(result_dict)

        # 确保最终返回数量足够（如果有重复会少于n，这里补充生成）
        while len(pred_result_list) < n:
            # 随机选择已有结果进行微小调整生成新结果
            if not pred_result_list:  # 处理列表为空的边缘情况
                # 创建基础结果结构
                base = {b_name: 1 for b_name in ball_name_list}
            else:
                base = np.random.choice(pred_result_list)

            new_result = {}
            for k, v in base.items():  # 修正点：从base中获取键值对
                # 增加微小随机扰动
                new_val = v + np.random.randint(-1, 2)
                # 确保号码在有效范围内
                if "red" in k:
                    new_result[k] = max(1, min(33, new_val))
                else:
                    new_result[k] = max(1, min(16, new_val))

            # 检查新结果是否已存在
            new_tuple = tuple(sorted(new_result.items()))
            if new_tuple not in seen:
                seen.add(new_tuple)
                pred_result_list.append(new_result)

        return pred_result_list[:n]
    else:
        red_preds, red_name_list = get_red_ball_predict_result(
            red_graph,
            red_sess,
            pred_key_d,
            predict_features,
            m_args["red_sequence_len"],
            m_args["windows_size"],
            n,  # 获取 n 个红球预测结果
        )
        blue_preds, blue_name_list = get_blue_ball_predict_result(
            blue_graph,
            blue_sess,
            pred_key_d,
            name,
            predict_features,
            m_args["blue_sequence_len"],
            m_args["windows_size"],
            n,  # 获取 n 个蓝球预测结果
        )
        ball_name_list = [
            "{}_{}".format(name[mode], i) for name, i in red_name_list
        ] + ["{}_{}".format(name[mode], i) for name, i in blue_name_list]

        # 生成并去重结果
        pred_result_list = []
        seen = set()
        for i in range(n):
            # 将蓝球整数转为列表（如 12 → [12]）
            pred_result = red_preds[i].tolist() + [blue_preds[i].item()]
            result_dict = {
                b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result)
            }
            result_tuple = tuple(sorted(result_dict.items()))
            if result_tuple not in seen:
                seen.add(result_tuple)
                pred_result_list.append(result_dict)

        # 确保最终返回数量足够
        while len(pred_result_list) < n:
            # 随机选择已有结果进行微小调整生成新结果
            if not pred_result_list:  # 处理列表为空的边缘情况
                # 创建基础结果结构
                base = {b_name: 1 for b_name in ball_name_list}
            else:
                base = np.random.choice(pred_result_list)

            new_result = {}
            for k, v in base.items():  # 修正点：从base中获取键值对
                new_val = v + np.random.randint(-1, 2)
                # 根据大乐透规则调整范围
                if "red" in k:
                    new_result[k] = max(1, min(35, new_val))
                else:
                    new_result[k] = max(1, min(12, new_val))

            # 检查新结果是否已存在
            new_tuple = tuple(sorted(new_result.items()))
            if new_tuple not in seen:
                seen.add(new_tuple)
                pred_result_list.append(new_result)

        return pred_result_list[:n]


def run(name, n=10):
    """执行预测"""
    try:
        red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number = (
            load_model(name)
        )
        windows_size = model_args[name]["model_args"]["windows_size"]
        data = spider(name, 1, current_number, "predict")
        next_period = int(current_number) + 1
        logger.info(
            f"\n===== 【{name_path[name]['name']}】预测{next_period}期 共{n}条预测结果汇总(随机干扰) ====="
        )
        predict_features_ = try_error(1, name, data.iloc[:windows_size], windows_size)
        results = get_final_result(
            red_graph,
            red_sess,
            blue_graph,
            blue_sess,
            pred_key_d,
            name,
            predict_features_,
            n=n,  # 获取多个预测结果
        )

        # 格式化输出结果
        for i, res in enumerate(results, 1):
            logger.info(f"第{i}组预测号码: {res}")

        # 关闭会话释放资源
        red_sess.close()
        blue_sess.close()
    except Exception as e:
        logger.exception("预测失败，错误信息：%s", e)  # 用 exception 会自动带 traceback

        try:
            red_sess.close()
            blue_sess.close()
        except:
            pass


if __name__ == "__main__":
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(args.name, args.num)
