from book_list_getter import BookListGetter, bookList, WordListGetter
from vector_methods.empath import EmpathReviewer
from vector_methods.tf_idf import TF_IDF_Reviewer
from vector_methods.doc2vec import Doc2VecReviewer
from vector_methods.word2vec import Word2VecReviewer
from vector_methods.glove import GloVeReviewer
import csv

book_list_getter = BookListGetter()

def test_wordList():
    wordListGetter = WordListGetter()
    print(wordListGetter.empath_word_list())

def test_empath():
    reviewer = EmpathReviewer()
    reviewer.base_review(book_list_getter.get_books([bookList.textbooks, bookList.goodreads, bookList.NSTA]))

def test_llm():
    reviewer = LLM_Reviewer()
    reviewer.conversation_review(book_list_getter.get_books([bookList.textbooks, bookList.goodreads, bookList.NSTA]))

def brief_llm():
    reviewer = LLM_Reviewer()
    reviewer.conversation_previewer(book_list_getter.get_books([bookList.textbooks, bookList.goodreads, bookList.NSTA]))

def doc2vecTest():
    reviewer = Doc2VecReviewer()
    reviewer.trained_review(book_list_getter.get_books([bookList.textbooks, bookList.goodreads, bookList.NSTA]))

def word2vecTest():
    reviewer = Word2VecReviewer()
    reviewer.trained_review(book_list_getter.get_books([bookList.textbooks, bookList.goodreads, bookList.NSTA]))

def GloVeTest():
    reviewer = GloVeReviewer()
    reviewer.trained_review(book_list_getter.get_books([bookList.textbooks, bookList.goodreads, bookList.NSTA]))


def save_book_list():
    book_list = book_list_getter.get_books([bookList.textbooks, bookList.goodreads, bookList.NSTA])
    csv_filename = "books.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ISBN_or_Title", "Summary", "Boolean"])
        for book in book_list:
            writer.writerow(book)
    print(f"✅ CSV file '{csv_filename}' written successfully!")


if __name__ == "__main__":
    brief_llm()



