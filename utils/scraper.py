import praw
import fs
import sys
from praw.models import MoreComments
import json 
import importlib.util

spec = importlib.util.spec_from_file_location("keys", "../config/keys.py")
keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keys)


class Scraper():
    def __init__(self): 
        self.reddit = praw.Reddit(client_id=keys.reddit_client_id,
                     client_secret=keys.reddit_client_secret,
                     user_agent='ChatJawn 0.1')
        self.text_string = ''
        self.threads = []
        self.posts = []

        # Test if read-only instance was created
        print(self.reddit.read_only)             

    def get_submissions(self, subreddit_name, limit):
        for submission in self.reddit.subreddit(subreddit_name).hot(limit=limit):
            post = { "title": submission.title, 
                      "comments": []}
            top_comments = list(submission.comments)
            for comment in top_comments: 
                # This is when you have to click "more comments" on Reddit
                if isinstance(comment, MoreComments):
                    continue             
                post["comments"].append(comment.body)
                temp = comment.body.replace("\n", "N_L")
                self.text_string += "\n\n" + temp
            self.posts.append(post)

    def get_comment_tree(self, subreddit_name, limit):
        for submission in self.reddit.subreddit(subreddit_name).hot(limit=limit):
            # self.text_string += submission.title.upper() + "\n\n"
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                if len(comment.replies) > 0: 
                    temp = []
                    self.text_string += comment.body.replace("\n", " ")
                    temp.append(comment.body.replace("\n", ""))
                    for reply in comment.replies:
                        self.text_string += reply.body.replace("\n", " ")
                        temp.append(reply.body.replace("\n", ""))
                    self.text_string += "$BREAK"
                    self.threads.append(temp) 

    def write_to_text_file(self): 
        with open("../data/comments/comments_long.txt", "a") as text_file:
            for thread in self.threads:
                write = "\n".join(thread)
                text_file.write(write)
                text_file.write("\n$BREAK\n")

        print("File written")

    def write_to_json(self):
        # Clean data
        for post in self.posts:
            for comment in post["comments"]: 
                comment.replace(",", "")
                comment.replace("\"", "")

        # convert list to json
        json_dump = json.dumps(self.posts)    

        with open("../data/posts.json", "w") as json_file:
            json_file.write(json_dump)


scrapy = Scraper()

subreddits = ['mildlyinteresting', 'todayilearned', 'pics', 'unpopularopinion', 'pcmasterace', 'technology', 'funny', 'space', 'dataisbeautiful', 'mademesmile', 'humansbeingbros', 'philadelphia', 'canada', 'explainitlikeimfive', 'blackpeopletwitter', 'twoxchromosomes']
long_subreddits = ['news', 'casualconversation', 'worldnews', 'politics', 'wallstreetbets', 'popular']

count = 0

if(len(sys.argv) > 1):
    if(sys.argv[1] == 'long'):
        for sub in long_subreddits: 
            count += 1
            if count > 0:
                print("Scraping -", sub)
                scrapy.get_comment_tree(sub, 3)
                scrapy.write_to_text_file()

else:
    for sub in subreddits: 
        count += 1
        if count > 3:
            print("Scraping -", sub)
            scrapy.get_comment_tree(sub, 15)
            scrapy.write_to_text_file()



 





