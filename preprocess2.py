import csv
import glob

if __name__ == '__main__':
    # stock = '0050'
    # stock = 'main'
    stock = '100000'
    filePath = 'dataset_main/{}.csv'.format(stock)

    allFilePath = '{}-all-close.csv'.format(stock)

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
        tempList.append(row[5])  # close
        # tempList[7] = row[5] # close
        # tempList[7] = row[3] # high
        # tempList[7] = row[3]  # low

        allFileWriter.writerow(tempList)
        tempList = row

    file.close()

    allFile.close()
