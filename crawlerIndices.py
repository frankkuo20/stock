import csv
import json

import os
import requests
import time


class Crawler:
    def __init__(self, stockNo, year, month):
        '''

        :param stockNo: XXXX
        :param year: XXXX
        :param month: XX
        '''
        self._stockNo = stockNo
        self._year = year
        self._month = month

    def run(self):
        stockNo = self._stockNo
        year = self._year
        month = self._month

        date = year + month + '01'

        time.sleep(10)
        url = 'http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json' \
              '&date={}&stockNo={}&_=1519473091902'.format(date, stockNo)
        req = requests.get(url)

        filePath = '{}/'.format(stockNo)
        if not os.path.exists(filePath):
            os.mkdir(filePath)

        # ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數"]
        head = ['date', 'volume', 'money', 'open', 'high', 'low', 'close', 'change', 'num']
        result = json.loads(req.text)
        fileName = '{}/{}-{}{}.csv'.format(stockNo, stockNo, year, month)
        csvFile = open(fileName, 'w', newline='')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(head)
        for row in result['data']:
            rowList = []
            rowList.append(row[0])
            rowList.append(row[1].replace(',', ''))
            rowList.append(float(row[2].replace(',', '')))
            rowList.append(float(row[3].replace(',', '')))
            rowList.append(float(row[4].replace(',', '')))
            rowList.append(float(row[5].replace(',', '')))
            rowList.append(float(row[6].replace(',', '')))
            change = row[7]
            if change == 'X0.00':
                change = '0'
            change = float(change)
            rowList.append(change)
            rowList.append(row[8].replace(',', ''))

            csvWriter.writerow(rowList)

        csvFile.close()
        print('{}...finish'.format(fileName))


if __name__ == '__main__':
    # month: 2003/06

    # stockNo = '0050'
    # startYear = 92 + 1911
    # year = str(startYear)
    #
    # for i in range(6, 13):
    #     month = '{0:02d}'.format(i)
    #     crawler = Crawler(stockNo, year, month)
    #     crawler.run()
    #
    # startYear = 93 + 1911
    # for j in range(startYear, 2018):
    #     year = str(j)
    #     for i in range(12):
    #         month = '{0:02d}'.format(i+1)
    #         crawler = Crawler(stockNo, year, month)
    #         crawler.run()

    # stockNo = '3008'
    # startYear = 91 + 1911
    # year = str(startYear)
    #
    # for i in range(3, 13):
    #     month = '{0:02d}'.format(i)
    #     crawler = Crawler(stockNo, year, month)
    #     crawler.run()
    #
    # startYear += 1
    # for j in range(startYear, 2018):
    #     year = str(j)
    #     for i in range(12):
    #         month = '{0:02d}'.format(i+1)
    #         crawler = Crawler(stockNo, year, month)
    #         crawler.run()
    #
    # year = '2018'
    # for i in range(1, 3):
    #     month = '{0:02d}'.format(i)
    #     crawler = Crawler(stockNo, year, month)
    #     crawler.run()

    stockNo = '2330'
    startYear = 83 + 1911
    year = str(startYear)

    for i in range(9, 13):
        month = '{0:02d}'.format(i)
        crawler = Crawler(stockNo, year, month)
        crawler.run()

    startYear += 1
    for j in range(startYear, 2018):
        year = str(j)
        for i in range(12):
            month = '{0:02d}'.format(i + 1)
            crawler = Crawler(stockNo, year, month)
            crawler.run()

    year = '2018'
    for i in range(1, 3):
        month = '{0:02d}'.format(i)
        crawler = Crawler(stockNo, year, month)
        crawler.run()

