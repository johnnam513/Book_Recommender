# Book Recommend System
This project is about book recommend system, considering the text data from Naver book information. TF-IDF, word2vec and doc2vec is used.

### How To Use
1. First you need to take book_id to do crawling. It is just okay to change the number, but in this project we only concerned with the bestsellers so we have to make independent crawler which takes book_id of bestsellers. Use take_book_id_list.py.
2. Next, crawl the book information by using crawler.py.
3. Finally, implement the models and find what you want to read!

### proposal.pptx
This presentation file deals with the whole outline of making book recommender system. It is written in Korean.

### Progress.pptx
This presentation file is used for progress presentation in 'Data Mining' course in Seoul National University.

### take_book__id_list.py
Can crawl the book_id of bestsellers. The result is in 'book_id_list'.

### crawler.py
Taek book informations from Naver. The result is in 'book_info'.

### book_info(total).zip
The preprocessed data, which is used when making the recommender, is in this zip file.
