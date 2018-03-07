import csv
import glob

if __name__ == '__main__':
    # stock = '0050'
    stock = 'main'
    filePath = 'dataset_main/{}.csv'.format(stock)

    allFilePath = '{}-all.csv'.format(stock)

    allFile = open(allFilePath, 'w', newline='')
    allFileWriter = csv.writer(allFile)

    head = ['date', 'volume', 'open', 'high', 'low', 'close', 'change', 'label']
    allFileWriter.writerow(head)

    tempList = []

    file = open(filePath, 'r')
    fileReader = csv.reader(file)
    next(fileReader)  # skip first line

    for row in fileReader:

        if not tempList:
            tempList = row
            continue

        tempList[7] = row[5]
        allFileWriter.writerow(tempList)
        tempList = row

    file.close()

    allFile.close()
