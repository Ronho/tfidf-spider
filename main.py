from tfidf import Tfidf
from spider import Spider

import nltk

spider = Spider(corpus='data', file_type='txt', language='german') 
spider.crawl()
spider.generate_term_index(additional_terms='CPU\nGPU\nStift\nGNU General Public License\nhäufig'.split('\n'))

tfidf = Tfidf(spider=spider, new_documents=spider.content, index_documents='overwrite')
print(tfidf)
print(tfidf.search('Ich unterstütze einige Typisierungen'))