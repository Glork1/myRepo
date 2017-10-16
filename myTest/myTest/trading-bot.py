import time
import sys, getopt
import datetime
from poloniex import poloniex

def main(argv):
	period = 10
	#pair = "BTC_USDT"
    #pair = "BTC_XMR"
	try:
		opts, args = getopt.getopt(argv,"hp:c:n:s:e:",["period=","currency="])
	except getopt.GetoptError:
		print("trading-bot.py -p <period length> -c <currency pair>")
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("trading-bot.py -p <period length> -c <currency pair>")
			sys.exit()
		elif opt in ("-p", "--period"):
			if (int(arg) in [300,900,1800,7200,14400,86400]):
				period = arg
			else:
				print("Poloniex requires periods in 300,900,1800,7200,14400, or 86400 second increments")
				sys.exit(2)
		elif opt in ("-c", "--currency"):
			pair = arg
	conn = poloniex("key pub to modify","key private to modify")
	while True:
            currentValues = conn.api_query("returnTicker")
            lastPairPrice = currentValues[pair]["last"]
            dataDate = datetime.datetime.now()
            print("{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())+" Period: %ss %s: %s" % (period,pair,lastPairPrice))
            time.sleep(int(period))

if __name__ == "__main__":
    main(sys.argv[1:])