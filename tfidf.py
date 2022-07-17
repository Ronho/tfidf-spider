import math
import numpy as np

from itertools import compress
from typing import Union, List

from spider import Spider

class Tfidf(object):
    """Term Frequency - Inverse Document Frequency
    """

    def __init__(
        self,
        spider:Spider,
        index_documents:str ='same',
        new_documents:Union[dict, None] = None
        
    ) -> None:
        """Constructor for Tfidf

        Args:
            spider (Spider): Spider object that will serve for initializing the
                inverse document frequency (idf)
            index_documents (str, optional): Three options:
                - same: spider corpus will serve as documents to be indexed 
                - new: parameter `new_documents` will serve as documents to be
                    indexed
                - add: in addition to the spider corpus `new_documents` will
                    serve as documents to be indexed
                Defaults to 'same'.
            new_documents (Union[dict, None], optional): In case
                `index_documents` is set to 'new' or to 'add' `new_documents'
                will be used for documents to be indexed. Defaults to None.

        Raises:
            Exception: Raised if index_documents is unknown.
        """

        self.spider = spider
        self.idf = {}
        self.indexed_documents = {}
        self.word_vector = []
        
        self.init_term_frequency()

        if index_documents == 'same':
            content = spider.content.copy()
        elif index_documents == 'new':
            content = new_documents
        elif index_documents == 'add':
            # files
            content = spider.content.copy()
            content.update(new_documents)
        else:
            raise Exception(f'Unknown input for index_documents:'
                f'{index_documents}.')

        self.insert_documents(content)

    def init_term_frequency(self) -> None:
        """Initialize term frequency

        Every idf value for each term of spiders index is calculated and stored.
        """
        term_index = self.spider.get_index_dictionary(sort=True)
        
        total_count = sum(term_index.values())
        self.word_vector = list(term_index.keys())
        
        for word in self.word_vector:
            self.idf[word] = math.log(total_count/term_index[word])

    def insert_documents(self, content:dict) -> None:
        """Inserting new documents that will be used as indexed documents

        A word vector is calculated and stored for each document of `content`.

        Args:
            content (dict):
                - key: document name
                - value: content of the document
        """
        for key, value in content.items():
            document_word_vector = self.create_word_vector(text=value)
            self.indexed_documents[key] = document_word_vector

    def create_word_vector(self, text:str) -> np.array:
        """Create a word vector for a specific text

        During the vectorization process it is checked whether every tokenized
        and stemmatized word from the text occurs in the stored word list.
        If so, the respective tfidf value is calculated and set. Otherwise, 0
        is set.

        The tfidf value is obtained by multiplying the tf and id values.

        Args:
            text (str): Text could be a single word, sentence or paragraph.

        Returns:
            np.array: Word vector.
        """
        new_spider = Spider(language=self.spider.language)
        new_spider.add_new_token(text)
        term_index = new_spider.get_index_dictionary()
        len_tokens = sum(term_index.values())
        new_spider.clear_index()

        # create document word vector
        document_word_vector = np.zeros(len(self.word_vector))
        for token in term_index.keys():
            if token in self.word_vector:
                position = self.word_vector.index(token)
                # penalty-or-booster * word rate
                document_word_vector[position] = self.idf[token] \
                    * (term_index[token]/len_tokens)
        
        return document_word_vector

    def get_idf_for_token(self, token:str) -> float:
        """Returns idf value for specific token

        Args:
            token (str): Single token of an already stemmanized word.

        Returns:
            float: Idf value for the specific token.
        """
        return self.idf[token]
        
    def get_token_for_vector(self, word_vector:np.array) -> List[str]:
        """Returns original word for a word vector

        Args:
            word_vector (np.array): Word vector for specific text.

        Returns:
            List[str]: Recognized tokens based on the word vector.
        """
        words = list(compress(self.word_vector, word_vector>0))
        return words

    def search(self, search_text:str, verbose:bool=True) -> str:
        """Searches the documents for a best match considering a search text

        First, the search text is converted to a word vector. Next, the search
        text word vector is compared to the word vector of all stored documents.
        
        Euclidean distance:
            d(p,q)^2 = (q_1 - p_1)^2 + (q_2 - p_2)^2 + ... + (q_n - p_n)^2

        Args:
            search_text (str): Text for which the most similar document is to be
                search for.
            verbose (bool, optional): If True, intermediate results will be
                printed. Defaults to True.

        Returns:
            str: Name of the document that matches the search string the best
                regarding the highest euclidean distance.
        """
        best_match_document = 'no document found'
        best_match_distance = 100
        
        search_text_vector = self.create_word_vector(search_text)
        if verbose:
            print(f'text vector: {search_text_vector}')
            print(f'recognized tokens: '
                f'{self.get_token_for_vector(search_text_vector)}')
        
        for key, value in self.indexed_documents.items():
            # Euclidean distance
            distance = np.linalg.norm(value-search_text_vector)
            if verbose:
                print(f'{distance:.5f} for doc {key}')
            if distance<best_match_distance:
                best_match_distance=distance
                best_match_document=key
        if verbose:
            print(f'closest doc is {best_match_document} with distance '
                f'{best_match_distance}')
        return best_match_document
        
    def __str__(self) -> str:
        """String representer for Tfidf objects

        Returns:
            str: Concatenated list of all document names.
        """
        document_names = ''
        for document_name in self.indexed_documents.keys():
            document_names += f'{document_name}, '
        return document_names[:-2]