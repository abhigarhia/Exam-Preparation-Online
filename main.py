import PyPDF2

Obj = open('notes.pdf', 'rb')
 
pdfReader = PyPDF2.PdfFileReader(Obj)
 
pages = pdfReader.numPages
 
text = []
for i in range(pages):

	pageObj = pdfReader.getPage(0)
 
	text.append(str((pageObj.extractText()).encode('cp1252')))

Obj.close()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(text)

row,col = X.shape[0],X.shape[1]
x = X.tocoo()

score = dict()

tokens = vectorizer.inverse_transform(X)
k = 0

for i in range(row):
	for j in range(col):
		
		if tokens[i][j] in score.keys():
			score[tokens[i][j]] += x.data[k]
		else:
			score[tokens[i][j]] = 1
		k += 1

score_value = []
score_key = sorted(score, key=score.get, reverse=True)
for i in score_key:
	score_value.append(score[i])
	
import pandas as pd
df = pd.DataFrame()
df['keyword'] = score_key
df['importance_value'] = score_value
df.to_csv('keyword.csv',index = False)	