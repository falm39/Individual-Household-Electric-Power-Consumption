import pandas as pd
from sklearn.preprocessing import Binarizer
from collections import defaultdict

# Veriyi yükleme
data = pd.read_csv('household_power_consumption.txt', sep=';', parse_dates=[[0, 1]], infer_datetime_format=True)

# Veriyi temizleme: Sadece sayısal sütunları alıyoruz
data = data.dropna()
data = data[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]

# Veriyi ikili hale getirme (Binarize)
scaler = Binarizer(threshold=0.1)
data_bin = scaler.fit_transform(data)

# Veriyi DataFrame formatına dönüştürme
data_bin = pd.DataFrame(data_bin, columns=data.columns)

# H-Mine algoritması için fonksiyon: Sık öğe kümelerini bulma
def h_mine(data_bin, min_support=0.01):
    # Adım 1: Verinin öğelerini bulma
    itemsets = defaultdict(list)
    for row_idx, row in data_bin.iterrows():
        for idx, val in row.items():
            if val == 1:
                itemsets[idx].append(row_idx)

    # Adım 2: 1 öğe kümelerini belirleme (ve destek hesaplama)
    frequent_itemsets = []
    for item, rows in itemsets.items():
        support = len(rows) / len(data_bin)
        if support >= min_support:
            frequent_itemsets.append((frozenset([item]), support))

    # Adım 3: Kümeleme ve kesişim işlemleri ile daha büyük öğe kümelerini bulma
    frequent_itemsets.sort(key=lambda x: x[1], reverse=True)
    
    result_itemsets = list(frequent_itemsets)  # Başlangıçta sadece 1 öğe kümelerini ekliyoruz
    k = 2

    # Adım 4: H-Mine algoritmasında bir sonraki kümeleri oluşturma
    while result_itemsets:
        current_itemsets = []
        for i in range(len(result_itemsets)):
            for j in range(i + 1, len(result_itemsets)):
                itemset1, support1 = result_itemsets[i]
                itemset2, support2 = result_itemsets[j]
                
                intersection = itemset1 & itemset2
                if len(intersection) == k - 1:  # Kesişimde k-1 öğe olmalı
                    # Kesişimdeki öğelerle yeni bir kümeyi oluştur
                    common_rows = set(itemsets[itemset1.pop()] & itemsets[itemset2.pop()])
                    new_support = len(common_rows) / len(data_bin)
                    if new_support >= min_support:
                        current_itemsets.append((intersection, new_support))
        
        result_itemsets = current_itemsets
        k += 1
    
    return frequent_itemsets

# H-Mine algoritmasını uygulama
frequent_itemsets = h_mine(data_bin, min_support=0.01)

# Sonuçları yazdırma
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets:
    print(f"Itemset: {itemset}, Support: {support}")
