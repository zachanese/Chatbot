import json
import requests
import re
import tensorflow as tf
import execute
from slackbot.bot import respond_to
from slackbot.bot import listen_to

sess = tf.Session()
sess, model, enc_vocab, rev_dec_vocab = execute.init_session(sess, conf='../seq2seq_serve.ini')

@respond_to('(*)', re.IGNORECASE)
def tf_reply(message):
    tf_msg = execute.decode_line(sess, model, enc_vocab, rev_dec_vocab, message.body)
    message.reply(tf_msg)
