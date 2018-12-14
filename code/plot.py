import matplotlib.pyplot as plt
import json
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plosts the log data after training")
    parser.add_argument("--path", "-p", type=str, default="./candy_john/log.txt",  help="Path to the log file")
    parser.add_argument("--title", "-t", type=str, default="Johnson Training")
    args = parser.parse_args()

    f = open(args.path, 'r+')

    lines = f.read().split('}')
    for i in range(len(lines)):
        lines[i] = lines[i]+"}"
        #print(l)

    _ = lines.pop()

    meta = lines.pop(0)
    meta = json.loads(meta)
    print(meta)
    for i in range(10):
        _ = lines.pop(0) #throw away the first 2 since no training has been done
    print(lines)

    time = []
    epoch = []
    iteration = []
    loss = []
    for l in lines:
        data = json.loads(l)

        time.append(data["time"])
        epoch.append(data["epoch"])
        iteration.append(data["interation"])
        loss.append(data["loss"])

    plt.plot(iteration, loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(args.title + ": Style Weight=" + str(meta["style_weight"]) + ", Content Weight=" + str(meta["content_weight"]) + ", LR=" + str(meta["learning_rate"]))

    plt.show()

    print(iteration)

    print(loss)