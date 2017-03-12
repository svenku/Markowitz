## Markowitz Brute Force Efficient Frontier Finder v1.0
## autor: Sven Kunsing
## (c) SK 2016

## kokkuvõte eestikeelsena:

##Antud programm küsib kasutajalt sisendina mõned olemasolevad USA equity tickerid,
##näiteks võib proovida sector index ETFide tickereid http://www.sectorspdr.com/sectorspdr/
##a la 'XLE XLU XLP' jne
##
##Nende tickerite kohta küsitakse yahoost hindade ajalugu viimase kolme aasta kohta. Periood on fiksitud
##ja tickerite küsimisega ei maksaks liialdada, sest Yahoo piirab miskitpidi neid päringuid.
##Täpselt piirangutega tutvumine jäi skoobist välja.
##
##Ajaloost arvutatakse kuutootlused, nende korrelatsioonid ja standardhälbed.
##
##Edasi lahendatakse portfelliriski optimeerimine brute force meetodil, genereerides 50k juhuslike
##kaaludega (siinkohal jäi testimiseks aega vajaka, et kas need ikka on päris juhulsikud kaalud...
##vist mitte... ootaks kommentaare/näpunäiteid, kuidas seda paremini teha...) portfelle ja arvutades neile
##riskid ja tootlused ning lisades kõik need portfolios listi (kust neid vajadusel leida).
##mulle tundub, et väga palju ei kiputa arvutama ekstreemseid kaale ja probleem süveneb tickerite hulga kasvades
##2-3 tickeriga on tulemus ootuspärasema väljanägemisega (kuna siis näib kaaluekstreemumite esinemine tõenäolisem)
##Proovisin ka miljoni portfelliga, aga mulle tundub, et see hakkas nagu kergelt mu arvutile üle jõu käima. Mälukasutust
##ma niiviisi jooksu pealt kohe hinnata ei oska. Kindlasti on selleks lisaks peast arvutamisele ka mingi süsteemne tugi.
##
##Edasi sorteeritakse baas tootluste järgi kahanevalt ja nopitakse järjest välja leitud parima riskiga
##portfellid.
##
##Kahjuks kulus numpy, pyploti ja pandas pakettidest minimaalse arusaamise omandamise ja tulemuse saavutamiseni
##nii palju aega, et lahendasin ainult miinimumprogrammi, mille endale olin seadnud.
##
##scipy-ga sama ülesande lahendamine elegantsemalt ja mäluga säästvamalt ümberkäimise proovimine jääb
##mingiks teiseks korraks.




import numpy as np
import matplotlib.pyplot as plt
import math
import pandas_datareader.data as web
import datetime
from sys import exit

## init 

port_risks = []
port_returns = []
port_weights = []
opt_portfolios = []
opt_risks = []
opt_returns = []
tickers = []

def getTickers():
    return input('Enter 2 to 10 valid US equity tickers separated by space: ').upper().split()

## get tickers from user

while len(tickers) < 2 or len(tickers) > 10:
    tickers = getTickers()

num_tickers = len(tickers)
tickers = sorted(tickers)

## use following hardcoded periods only (no user control), hope it works, bc
## see here https://developer.yahoo.com/yql/guide/usage_info_limits.html

start = datetime.datetime(2013,12,31)
end = datetime.datetime(2016,12,31)

## get yahoo financial 'adj close' data (daily), resample to monthly, calc monthly returns

f = web.DataReader(tickers, 'yahoo', start, end)['Adj Close']
m_data = f.resample('M').last()
m_returns = m_data.pct_change()

## check & display data span (just in case yahoo limits queries)

begin_data_date = m_data.index.tolist()[0].date()
end_data_date = m_data.index.tolist()[-1].date()
total_months = (end_data_date.year - begin_data_date.year)*12 + end_data_date.month - begin_data_date.month

print('Data should run from  ' + str(start.date()) + ' to ' +  str(end.date()))
print('It actually runs from ' + str(begin_data_date) + ' to ' +  str(end_data_date))
print('Calculations are based on ' + str(total_months) + ' monthly adjusted closing prices.')

## check the sanity of the user, exit if not 'y' entered
## rather unnecessary, was included to have an if statement as requested by requirements of the task

ans = input('Are you sure you want to continue? (Enter "y" to continue): ').lower()

if ans != 'y':
    exit()

## calculate monthly (not annualized!) returns, corrs and risks(monthly return stdevs)

returns = np.power(m_data.tail(1).values/m_data.head(1).values, 1/36)-1
correlations = m_returns.corr().values
stdevs = np.array(m_returns.std(axis=0).tolist())

## diag matrix of stdevs

stdevs_diag = np.eye(num_tickers) * stdevs

## find covariance matrix

covariations = np.dot(stdevs_diag, np.dot(correlations, stdevs_diag))

## generate silly number of portfolios

for i in range(50000):

    ## random weights vector and transpose it
    weights = np.random.random(num_tickers)
    weights /= weights.sum()

    weights_col = weights.reshape(num_tickers,1)

    ## calculate each portfolio's risk and return, put them to separate lists, also returns
    p_risk = math.sqrt(np.dot(weights, np.dot(covariations, weights_col)))
    p_return = np.dot(returns, weights)

    port_risks.append(p_risk)
    port_returns.append(p_return)
    port_weights.append(weights.tolist())

## zip the list together - so that in later apps one can locate specific portfolios

portfolios = list(zip(port_risks, port_returns, port_weights))

## sort them by return from high to low

portfolios = sorted(portfolios, key = lambda x: x[1], reverse = True)

## first in list is definitely part of efficient frontier

opt_portfolios.append(portfolios[0])
opt_risks.append(portfolios[0][0])
opt_returns.append(portfolios[0][1])

min_risk = portfolios[0][0]

## select rest of optimal portfolios (= efficient frontier).

for el in portfolios:
    if el[0] < min_risk:
        opt_portfolios.append(el)
        opt_risks.append(el[0])
        opt_returns.append(el[1])
        min_risk = el[0]
    
## define and plot all the portfolios and efficient frontier

plt.scatter(port_risks, port_returns, c='blue')
plt.plot(opt_risks, opt_returns, linewidth = 4.0, c='red')
plt.xlabel('risk, monthly')
plt.ylabel('return, monthly')
plt.title('All (blue) and Optimal (red) Portfolios on Risk-Return plane')

plt.show()
