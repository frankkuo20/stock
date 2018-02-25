import csv
import glob

if __name__ == '__main__':
    filePaths = glob.glob('0050/*')

    allFilePath = '0050-all.csv'

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
            if index == 0:
                tempList = row
                continue

            tempList.append(row[6])
            allFileWriter.writerow(tempList)
            tempList = row

        file.close()

    allFile.close()
