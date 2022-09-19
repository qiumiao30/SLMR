from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from pickle import dump
import numpy as np
import pandas as pd

from args import get_parser


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(md['num_values'],
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """

    if dataset == "SMD":
        dataset_folder = "datasets/ServerMachineDataset"
        output_folder = "datasets/ServerMachineDataset/processed"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )

    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/data"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)
        print(dataset, "test_label", labels.shape)

        with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)

    elif dataset == "SWAT":
        dataset_folder = "datasets/SWAT/"
        output_folder = "datasets/SWAT/processed"
        makedirs(output_folder, exist_ok=True)
        train_df = pd.read_csv(dataset_folder+"SWaT_Dataset_Normal_v0.csv", delimiter=",")
        #train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
        #th_time = train_df['Timestamp'][0] + pd.Timedelta('4H')
        #train_df = train_df[train_df['Timestamp'] > th_time]
        # train_df["Timestamp"] = train_df["Timestamp"].replace("Timestamp", " Timestamp")
        # train_df["Normal/Attack"] = train_df["Normal/Attack"].replace("Attack", "A ttack")
        train_df_ = train_df.drop([" Timestamp", "Normal/Attack"], axis=1)
        for i in list(train_df_):
            train_df_[i] = train_df_[i].apply(lambda x: str(x).replace(",", "."))
        train_df_ = train_df_.astype(float)
        X_train = train_df_.values
        with open(path.join(output_folder, dataset + "_" + "train" + ".pkl"), "wb") as file:
            dump(X_train, file)

        test_df = pd.read_csv(dataset_folder+"SWaT_Dataset_Attack_v0.csv", delimiter=",")
        test_df_ = test_df.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        for i in list(test_df_):
            test_df_[i] = test_df_[i].apply(lambda x: str(x).replace(",", "."))
        test_df_ = test_df_.astype(float)
        X_test = test_df_.values
        with open(path.join(output_folder, dataset + "_" + "test" + ".pkl"), "wb") as file:
            dump(X_test, file)

        y_test = []
        for index in test_df['Normal/Attack'].index:
            label = test_df['Normal/Attack'].get(index)
            if label == "Normal":
                y_test.append(0)
            elif label == "Attack":
                y_test.append(1)
        y_test = np.asarray(y_test)
        with open(path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(y_test, file)

    elif dataset == "WADI":
        dataset_folder = "datasets/WADI/"
        output_folder = "datasets/WADI/processed"
        makedirs(output_folder, exist_ok=True)
        train_df = pd.read_csv(dataset_folder+"WADI_14days_new.csv", delimiter=",")
        train_df_ = train_df.drop(["Date", "Time", "2_LS_001_AL", "2_LS_002_AL", "2_P_001_STATUS", "2_P_002_STATUS"], axis=1)
        for i in list(train_df_):
            train_df_[i] = train_df_[i].apply(lambda x: str(x).replace(",", "."))
        train_df_ = train_df_.astype(float)
        X_train = train_df_.values
        with open(path.join(output_folder, dataset + "_" + "train" + ".pkl"), "wb") as file:
            dump(X_train, file)

        test_df = pd.read_csv(dataset_folder+"WADI_attackdataLABLE.csv", delimiter=",")
        test_df_ = test_df.drop(["Date ", "Time", "2_LS_001_AL", "2_LS_002_AL", "2_P_001_STATUS", "2_P_002_STATUS",
                                 "Attack LABLE (1:No Attack, -1:Attack)"], axis=1)
        for i in list(test_df_):
            test_df_[i] = test_df_[i].apply(lambda x: str(x).replace(",", "."))
        test_df_ = test_df_.astype(float)
        X_test = test_df_.values
        with open(path.join(output_folder, dataset + "_" + "test" + ".pkl"), "wb") as file:
            dump(X_test, file)

        y_test = []
        for index in test_df['Attack LABLE (1:No Attack, -1:Attack)'].index:
            label = test_df['Attack LABLE (1:No Attack, -1:Attack)'].get(index)
            if label == 1:
                y_test.append(0)
            elif label == -1:
                y_test.append(1)
        y_test = np.asarray(y_test)
        with open(path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(y_test, file)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)
