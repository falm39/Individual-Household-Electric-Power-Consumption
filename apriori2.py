import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import Binarizer

# Veriyi yükleme
data = pd.read_csv('household_power_consumption.txt', sep=';', parse_dates=[[0, 1]], infer_datetime_format=True)

# Veriyi temizleme: Sadece sayısal sütunları alıyoruz
data = data.dropna()
data = data[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]

# Veriyi ikili hale getirme (Binarize)
# Binarizer kullanarak değerleri 0 veya 1'e dönüştürüyoruz (örneğin: belirli bir eşik değeri üzerinden)
scaler = Binarizer(threshold=0.1)
data_bin = scaler.fit_transform(data)

# Veriyi DataFrame formatına dönüştürme
data_bin = pd.DataFrame(data_bin, columns=data.columns)

# Apriori algoritmasını uygulama
frequent_itemsets = apriori(data_bin, min_support=0.01, use_colnames=True)

# Association rules oluşturma
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1, num_itemsets=len(frequent_itemsets))

# Sonuçları yazdırma
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)
