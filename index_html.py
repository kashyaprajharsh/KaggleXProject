css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #144272
}
.chat-message.bot {
    background-color: #0A2647
}
.chat-message .avatar {
  width: 20%;
}
.source-message.source{
  text-align: right
  width 50%
  padding: 0 1.5rem;
  color: #fff

}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 100%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

source_template='''
<div class="source-message source">
  {{MSG}}
</div>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/ZffD9pB/digital-painting-mountain-with-colorful-tree-foreground.jpg" alt="digital-painting-mountain-with-colorful-tree-foreground">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
     <img src="https://i.ibb.co/v4JktLz/milford-sound-fiordland-new-zealand.jpg" alt="milford-sound-fiordland-new-zealand">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''