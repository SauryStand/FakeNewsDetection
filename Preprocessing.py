# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行

def generateRawCSV(filename):

    with open(filename,'r', encoding='utf-8') as f:
        lines = f.readlines()
        lineList = []
        lines.remove(lines[0])
        for line in lines:
            line = line.strip().split("\t")

            lineList.append(line)

            #source.write(line[0]+'\n')
            #target.write(line[1]+'\n')
    #tweetId	tweetText	userId	imageId(s)	username	timestamp	label
    columns = ['tweetId','tweetText','userId', 'imageId(s)', 'username', 'timestamp', 'label']
    df = pd.DataFrame(lineList, columns=columns)

    print(df.head())

    df.to_csv('dataset/raw_training.csv')

#source.close()
#target.close()

if __name__ == '__main__':
    filename = 'dataset/mediaeval-2015-trainingset.txt'
    generateRawCSV(filename)