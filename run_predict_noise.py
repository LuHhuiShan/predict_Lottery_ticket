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
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", default="ssq", type=str, help="选择训练数据: 双色球/大乐透"
)
parser.add_argument(
    "--num_preds",
    default=10,
    type=int,
    help="生成预测结果的数量（默认10条）",
)
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


def add_controlled_noise(features, name):
    """
    为输入特征添加可控随机扰动（保证结果不同但不偏离合理范围）

    Args:
        features: 原始输入特征（DataFrame）
        name: 玩法类型（"ssq" / "dlt"）
        noise_scale: 噪声强度（默认 0.1，可按需调整）

    Returns:
        noisy_features: 加噪后的特征 DataFrame
    """
    # 复制原始特征避免修改源数据
    noisy_features = features.copy()

    # ========= 1. 根据玩法确定红/蓝球列名 =========
    if name == "ssq":
        # 双色球：6 红 1 蓝
        red_cols = [f"红球_{i}" for i in range(1, 7)]
        blue_cols = ["蓝球"]
        red_max = 33
        blue_max = 16
        # 双色球（6红1蓝）：
        #   对于红球，noise_scale 大约为 0.05 - 0.15 之间（这样能保证生成的红球不偏离合理的范围，同时能增加一定的随机性）。
        #   对于蓝球，noise_scale 可以设置为 0.1 - 0.3，因为蓝球的范围较小，噪声过大会导致蓝球超出范围。
        red_noise_scale = 0.3  # 红球噪声强度
        blue_noise_scale = 0.3  # 蓝球噪声强度
    elif name == "dlt":
        # 大乐透：5 红 2 蓝
        red_cols = [f"红球_{i}" for i in range(1, 6)]
        blue_cols = [f"蓝球_{i}" for i in range(1, 3)]
        red_max = 35
        blue_max = 12
        # 大乐透（5红2蓝）：
        #   对于红球，noise_scale 大约为 0.05 - 0.1，因为红球的范围更大（1-35），噪声可以适度增加。
        #   对于蓝球，noise_scale 可以设置为 0.1 - 0.3，与双色球类似。
        red_noise_scale = 0.08  # 红球噪声强度
        blue_noise_scale = 0.2  # 蓝球噪声强度
    else:
        raise ValueError(f"不支持的玩法类型：{name}")

    # 实际存在于 DataFrame 中的列（防止有些列名不存在时报错）
    target_cols = [c for c in red_cols + blue_cols if c in noisy_features.columns]

    # ========= 2. 对目标列统一转换为数值类型 =========
    for col in target_cols:
        # 如果原始数据是 '02' 这种字符串，这里会转成 2
        noisy_features[col] = pd.to_numeric(noisy_features[col], errors="raise")

    # ========= 3. 为每一列添加随机扰动 =========
    for col in target_cols:
        # 选择不同的噪声强度
        if col in red_cols:
            noise_scale = red_noise_scale
        else:  # 蓝球列
            noise_scale = blue_noise_scale

        # 生成微小随机噪声（-k ~ +k），乘以 noise_scale 控制强度
        # 这里先生成连续值，再四舍五入为整数噪声
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(noisy_features))
        noise_int = np.round(noise).astype(int)

        # 加上扰动
        noisy_col = noisy_features[col].astype(int) + noise_int

        # 根据红/蓝球范围做裁剪
        if col in red_cols:
            max_val = red_max
        else:  # 蓝球列
            max_val = blue_max

        noisy_col = np.clip(noisy_col, 1, max_val)

        # 保存回 DataFrame，保持为整数
        noisy_features[col] = noisy_col.astype(int)

    return noisy_features


def get_red_ball_predict_result(
    red_graph, red_sess, pred_key_d, predict_features, sequence_len, windows_size
):
    """获取红球预测结果"""
    name_list = [(ball_name[0], i + 1) for i in range(sequence_len)]
    data = (
        predict_features[
            ["{}_{}".format(name[0], i) for name, i in name_list]
        ].values.astype(int)
        - 1
    )
    with red_graph.as_default():
        reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(
            pred_key_d[ball_name[0][0]]
        )
        pred = red_sess.run(
            reverse_sequence,
            feed_dict={
                "inputs:0": data.reshape(1, windows_size, sequence_len),
                "sequence_length:0": np.array([sequence_len] * 1),
            },
        )
    return pred, name_list


def get_blue_ball_predict_result(
    blue_graph,
    blue_sess,
    pred_key_d,
    name,
    predict_features,
    sequence_len,
    windows_size,
):
    """获取蓝球预测结果"""
    if name == "ssq":
        data = predict_features[[ball_name[1][0]]].values.astype(int) - 1
        with blue_graph.as_default():
            softmax = tf.compat.v1.get_default_graph().get_tensor_by_name(
                pred_key_d[ball_name[1][0]]
            )
            pred = blue_sess.run(
                softmax, feed_dict={"inputs:0": data.reshape(1, windows_size)}
            )
        return pred
    else:
        name_list = [(ball_name[1], i + 1) for i in range(sequence_len)]
        data = (
            predict_features[
                ["{}_{}".format(name[0], i) for name, i in name_list]
            ].values.astype(int)
            - 1
        )
        with blue_graph.as_default():
            reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(
                pred_key_d[ball_name[1][0]]
            )
            pred = blue_sess.run(
                reverse_sequence,
                feed_dict={
                    "inputs:0": data.reshape(1, windows_size, sequence_len),
                    "sequence_length:0": np.array([sequence_len] * 1),
                },
            )
        return pred, name_list


def get_final_result(
    red_graph,
    red_sess,
    blue_graph,
    blue_sess,
    pred_key_d,
    name,
    predict_features,
    mode=0,
):
    """ " 最终预测函数"""
    m_args = model_args[name]["model_args"]
    if name == "ssq":
        red_pred, red_name_list = get_red_ball_predict_result(
            red_graph,
            red_sess,
            pred_key_d,
            predict_features,
            m_args["sequence_len"],
            m_args["windows_size"],
        )
        blue_pred = get_blue_ball_predict_result(
            blue_graph,
            blue_sess,
            pred_key_d,
            name,
            predict_features,
            0,
            m_args["windows_size"],
        )
        ball_name_list = [
            "{}_{}".format(name[mode], i) for name, i in red_name_list
        ] + [ball_name[1][mode]]
        pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
        return {
            b_name: int(res) + 1
            for b_name, res in zip(ball_name_list, pred_result_list)
        }
    else:
        red_pred, red_name_list = get_red_ball_predict_result(
            red_graph,
            red_sess,
            pred_key_d,
            predict_features,
            m_args["red_sequence_len"],
            m_args["windows_size"],
        )
        blue_pred, blue_name_list = get_blue_ball_predict_result(
            blue_graph,
            blue_sess,
            pred_key_d,
            name,
            predict_features,
            m_args["blue_sequence_len"],
            m_args["windows_size"],
        )
        ball_name_list = [
            "{}_{}".format(name[mode], i) for name, i in red_name_list
        ] + ["{}_{}".format(name[mode], i) for name, i in blue_name_list]
        pred_result_list = red_pred[0].tolist() + blue_pred[0].tolist()
        return {
            b_name: int(res) + 1
            for b_name, res in zip(ball_name_list, pred_result_list)
        }


def run(name, num_preds=10):
    """执行预测（支持生成多条不同结果）
    Args:
        name: 玩法类型（ssq/dlt）
        num_preds: 生成预测结果的数量
    """
    try:
        # 只加载一次模型（避免重复加载耗时）
        red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number = (
            load_model(name)
        )
        windows_size = model_args[name]["model_args"]["windows_size"]
        data = spider(name, 1, current_number, "predict")
        predict_period = int(current_number) + 1
        logger.info(f"【{name_path[name]['name']}】最近一期:{current_number}")
        logger.info(f"【{name_path[name]['name']}】预测期号：{predict_period}")
        logger.info(f"开始生成{num_preds}条不同预测结果...\n")

        # 获取基础输入特征（后续每次预测仅添加不同噪声）
        base_features = try_error(1, name, data.iloc[:windows_size], windows_size)

        # 循环生成num_preds条结果
        all_results = []
        for i in range(num_preds):
            # 为基础特征添加随机扰动（保证结果不同）
            noisy_features = add_controlled_noise(base_features, name)

            # 执行预测
            result = get_final_result(
                red_graph,
                red_sess,
                blue_graph,
                blue_sess,
                pred_key_d,
                name,
                noisy_features,
            )

            # ---- 关键修改：按 result 的中文键取值 ----
            if name == "ssq":
                # 双色球：6 红球，1 蓝球
                red_nums = [result[f"红球_{j+1}"] for j in range(6)]
                blue_nums = [result["蓝球"]]
            else:
                # 大乐透：5 红球，2 蓝球（假设 get_final_result 返回的键为：红球_1..5，蓝球_1, 蓝球_2）
                red_nums = [result[f"红球_{j+1}"] for j in range(5)]
                blue_nums = [result[f"蓝球_{j+1}"] for j in range(2)]

            # 格式化结果（更易读）
            formatted_result = {
                "预测序号": i + 1,
                "预测期号": predict_period,
                "红球": red_nums,
                "蓝球": blue_nums,
            }
            all_results.append(formatted_result)

            if len(all_results) == num_preds:
                # 最终汇总打印
                logger.info(
                    f"\n===== 【{name_path[name]['name']}】预测{predict_period}期 共{num_preds}条预测结果汇总(噪点干扰) ====="
                )
                for res in all_results:
                    logger.info(
                        f"第{res['预测序号']}条：红球{sorted(res['红球'])} | 蓝球{sorted(res['蓝球'])}"
                    )

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
        # 传入预测数量（默认10条，可通过命令行参数--num_preds修改）
        run(args.name, num_preds=args.num_preds)
