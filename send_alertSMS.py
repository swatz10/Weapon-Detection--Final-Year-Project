from twilio.rest import Client

# Find these values at https://twilio.com/user/account
account_sid = "AC51dcf85dfa87366655ecf225d6230023"
auth_token = "67e8115c2a2b0ec1c0c7e425aa90ffca"

client = Client(account_sid, auth_token)

client.api.account.messages.create(
    to="+917892416752",
    from_="+18443873970",
    body="alert !!!")