// SimpleML_EA.mq5
// Minimal EA: Only executes trades as instructed by ML backend
#property copyright "PropFirmTools"
#property version   "1.0"
#property strict

#include <Trade/Trade.mqh>
CTrade trade;

input string SignalURL = "http://127.0.0.1:5000/trade_signal"; // ML backend trade signal endpoint
input int    CheckInterval = 1; // Seconds between checks
input int    MagicNumber = 20250824;
input string TradeComment = "ML_EA";

// No duplicate/logic checks, just execute what ML says
void OnTick() {}

int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   EventSetTimer(CheckInterval);
   Print("SimpleML_EA initialized.");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("SimpleML_EA stopped.");
}

// Parse a single trade object from JSON-ish string
bool ParseSingleTrade(const string &json, int start_pos,
                      string &symbol, double &entry, double &sl, double &tp, double &lot, bool &isBuy)
{
   int pos, s, e;
   string temp;

   pos = StringFind(json, "\"symbol\":", start_pos);
   if(pos < 0) return false;
   s = StringFind(json, "\"", pos + 9);
   if(s < 0) return false;
   s++;
   e = StringFind(json, "\"", s);
   if(e < 0 || e <= s) return false;
   symbol = StringSubstr(json, s, e - s);

   pos = StringFind(json, "\"entry_price\":", e);
   if(pos < 0) return false;
   s = StringFind(json, ":", pos) + 1;
   if(s <= 0) return false;
   e = StringFind(json, ",", s);
   if(e < 0) e = StringFind(json, "}", s);
   if(e < 0 || e <= s) return false;
   temp = StringSubstr(json, s, e - s);
   StringTrimLeft(temp); StringTrimRight(temp);
   entry = StringToDouble(temp);

   pos = StringFind(json, "\"sl_price\":", e);
   if(pos < 0) return false;
   s = StringFind(json, ":", pos) + 1;
   if(s <= 0) return false;
   e = StringFind(json, ",", s);
   if(e < 0) e = StringFind(json, "}", s);
   if(e < 0 || e <= s) return false;
   temp = StringSubstr(json, s, e - s);
   StringTrimLeft(temp); StringTrimRight(temp);
   sl = StringToDouble(temp);

   pos = StringFind(json, "\"tp_price\":", e);
   if(pos < 0) return false;
   s = StringFind(json, ":", pos) + 1;
   if(s <= 0) return false;
   e = StringFind(json, ",", s);
   if(e < 0) e = StringFind(json, "}", s);
   if(e < 0 || e <= s) return false;
   temp = StringSubstr(json, s, e - s);
   StringTrimLeft(temp); StringTrimRight(temp);
   tp = StringToDouble(temp);

   pos = StringFind(json, "\"lot_size\":", e);
   if(pos < 0) return false;
   s = StringFind(json, ":", pos) + 1;
   if(s <= 0) return false;
   e = StringFind(json, ",", s);
   if(e < 0) e = StringFind(json, "}", s);
   if(e < 0 || e <= s) return false;
   temp = StringSubstr(json, s, e - s);
   StringTrimLeft(temp); StringTrimRight(temp);
   lot = StringToDouble(temp);

   pos = StringFind(json, "\"is_buy\":", e);
   if(pos < 0) return false;
   s = StringFind(json, ":", pos) + 1;
   if(s <= 0) return false;
   temp = StringSubstr(json, s, 5);
   StringTrimLeft(temp); StringTrimRight(temp);
   isBuy = (StringFind(temp, "true") >= 0);

   return true;
}

void OnTimer()
{

   uchar data[];
   uchar result[];
   string result_headers = "";
   if(WebRequest("GET", SignalURL, "", 2000, data, result, result_headers) != 200)
      return;

   string json = CharArrayToString(result);
   int pos = StringFind(json, "{");
   if(pos < 0) return;

   string symbol;
   double entry, sl, tp, lot;
   bool isBuy;
   if(!ParseSingleTrade(json, pos, symbol, entry, sl, tp, lot, isBuy))
      return;

   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   entry = NormalizeDouble(entry, digits);
   sl    = NormalizeDouble(sl, digits);
   tp    = NormalizeDouble(tp, digits);

   bool placed = false;
   if(isBuy)
      placed = trade.Buy(lot, symbol, entry, sl, tp, TradeComment);
   else
      placed = trade.Sell(lot, symbol, entry, sl, tp, TradeComment);

   Print("ML signal executed: ", symbol, " ", (isBuy ? "BUY" : "SELL"), " lot=", lot, " entry=", entry, " sl=", sl, " tp=", tp, " result=", placed);
}
