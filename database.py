import time
import sys, getopt
import datetime
import sqlite3
from poloniex import poloniex

def main(argv):
    #period = 10
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
    conn = poloniex("public key","private key")
    
    # Creation database
    
    connectionDB = sqlite3.connect("data.db")
    cursorDB = connectionDB.cursor()
    cursorDB.execute("""DROP TABLE dataByDate;""") # to delete
    sql_command = """
        CREATE TABLE IF NOT EXISTS dataByDate ( 
            myId INTEGER, 
            myName VARCHAR(20), 
            myBaseVolume DOUBLE, 
            myHighDay DOUBLE, 
            myHighestBid DOUBLE,
            myStatusFrozen DOUBLE,
            myLast DOUBLE,
            myLowDay DOUBLE,
            myLowestAsk DOUBLE,
            myPercentChange DOUBLE, 
            myDate DATE,
            PRIMARY KEY(myId,myDate));"""
    cursorDB.execute(sql_command)
    # Remplissage base de donnees
    #while True:
    for i in range(1):
        currentValues = conn.api_query("returnTicker") # <- this is a dictionary
        for key,value in currentValues.items():
            format_str = format_str = """
            INSERT INTO dataByDate (myId, myName, myBaseVolume, myHighDay, myHighestBid, myStatusFrozen, myLast, myLowDay, myLowestAsk, myPercentChange, myDate)
            VALUES ("{theId}", "{theName}", "{theBaseVolume}", "{theHighDay}","{theHighestBid}","{theStatusFrozen}","{theLast}","{theLowDay}","{theLowestAsk}","{thePercentChange}","{theDate}");"""
            sql_command = format_str.format(theId=value.get("id"), theName = key, theBaseVolume = value.get("baseVolume"),theHighDay = value.get("high24hr"), theHighestBid = value.get("highestBid"), theStatusFrozen = value.get("isFrozen"), theLast = value.get("last"), theLowDay = value.get("low24h"), theLowestAsk = value.get("lowestAsk"), thePercentChange = value.get("percentChange"),theDate =datetime.datetime.now())
            cursorDB.execute(sql_command)
        print(" Insert Done !")
        #time.sleep(int(period))
    # Test fetch
    cursorDB.execute("SELECT * FROM dataByDate") 
    print("fetchall:")
    result = cursorDB.fetchall() 
    for r in result:
        print(r)
    cursorDB.execute("SELECT * FROM dataByDate") 
    print("\nfetch one:")
    res = cursorDB.fetchone() 
    print(res)
    time.sleep(100)

    connectionDB.commit()
    cursorDB.close()
    connectionDB.close()

if __name__ == "__main__":
    main(sys.argv[1:])
