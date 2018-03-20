import csv
import glob

if __name__ == '__main__':
    # stock = '0050'
    stock = '3008'
    filePaths = glob.glob('dataset/{}/*'.format(stock))

    allFilePath = '{}-all.csv'.format(stock)

    allFile = open(allFilePath, 'w', newline='')
    allFileWriter = csv.writer(allFile)

    head = ['date', 'volume', 'money', 'open', 'high', 'low', 'close', 'change', 'num', 'label']
    allFileWriter.writerow(head)

    tempList = []

    for index, filePath in enumerate(filePaths):
        file = open(filePath, 'r')
        fileReader = csv.reader(file)
        next(fileReader)  # skip first line

        for row in fileReader:
            if not tempList:
                tempList = row
                continue

            tempList.append(row[6])
            allFileWriter.writerow(tempList)
            tempList = row

        file.close()

    allFile.close()
