import imaplib
import smtplib 
from email.mime.text import MIMEText 
import requests 
from icecream import ic 

# * email creds
EMAIL = '' 
PASSWORD = '' 
IMAP_SERVER = '' 
SMTP_SERVER = '' 

# connect to email server 
mail = imaplib.IMA4_SSL(IMAP_SERVER)
mail.login(EMAIL, PASSWORD)
mail.select('inbox')

# * search for all mails 
status, messages = mail.search(None, 'ALL')
email_ids = messages[0].split()

for email_id in email_ids: 
    status, msg_data = mail.fetch(email_id, '(RFC822)')
    msg = msg_data[0][1].decode('utf-8')

    # * extract email content, subject, requestedItems,, isAllowedDomain
