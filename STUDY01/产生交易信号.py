from K_get import Api


if __name__ == '__main__':
    api=Api()
    data=api.get_standard_futures_data()
    CUR_MACD=data['MACD_HIST']
    if CUR_MACD.iloc[-1]*CUR_MACD.iloc[-2]<0: #趋势反转捕捉信号
        #TODO:过滤假信号
        pass

        if CUR_MACD.iloc[-1]>0:
            print("多头头寸建仓")
        else:
            print("空头头寸建仓")
    else :
        print(f"当前macd_hist:{data['MACD_HIST'].iloc[-1]}")