from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from models.bert.load_bert_summarizer import load_bert_summarizer
from models.bert.preprocess_text import preprocess_lecture_text, setup_nltk
from models.bert.chunk_text import create_smart_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertSummarizer:
    def __init__(
        self,
        model_name: str = "philschmid/bart-large-cnn-samsum",
        chunk_size: int = 800,
        overlap_size: int = 100,
        max_summary_ratio: float = 0.3,
    ):
        setup_nltk()
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_summary_ratio = max_summary_ratio
        self.model = load_bert_summarizer(model_name)

    def _summarize_chunk(
        self,
        chunk: Dict[str, str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Summarize a single chunk while considering context.
        """
        try:
            full_text = chunk["text"]
            if chunk["next_context"]:
                full_text += f" {chunk['next_context']}"

            text_words = len(word_tokenize(full_text))
            if not min_length:
                min_length = min(50, max(30, int(text_words * 0.1)))
            if not max_length:
                max_length = min(150, max(100, int(text_words * self.max_summary_ratio)))

            summary = self.model(
                full_text,
                min_length=min_length,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )[0]["summary_text"].strip()

            return summary
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return chunk["text"][:200] + "..."

    def process_lecture(
        self,
        text: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Process and summarize a complete lecture transcript.
        """
        try:
            clean_text = preprocess_lecture_text(text)
            if not clean_text:
                return {
                    "error": "Empty or invalid text after preprocessing",
                    "detailed_summary": "",
                    "brief_summary": "",
                    "key_points": [],
                }

            chunks = create_smart_chunks(clean_text, self.chunk_size, self.overlap_size)
            chunk_summaries = []
            for chunk in tqdm(chunks, desc="Processing lecture chunks"):
                summary = self._summarize_chunk(chunk, min_length, max_length)
                chunk_summaries.append(summary)

            detailed_summary = " ".join(chunk_summaries)

            try:
                brief_summary = self.model(
                    detailed_summary,
                    min_length=50,
                    max_length=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )[0]["summary_text"]
            except Exception as e:
                logger.error(f"Error creating brief summary: {str(e)}")
                brief_summary = chunk_summaries[0]

            key_points = self._extract_key_points(detailed_summary)

            return {
                "error": None,
                "detailed_summary": detailed_summary,
                "brief_summary": brief_summary,
                "key_points": key_points,
            }
        except Exception as e:
            logger.error(f"Error processing lecture: {str(e)}")
            return {
                "error": f"Processing failed: {str(e)}",
                "detailed_summary": "",
                "brief_summary": "",
                "key_points": [],
            }

    def _extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key points from the summary."""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= num_points:
                return sentences

            indices = np.linspace(0, len(sentences) - 1, num_points, dtype=int)
            return [sentences[i] for i in indices]
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return []
        
if __name__ == "__main__":
    summarizer = BertSummarizer()

    # Test with sample lecture text
    lecture_text = """Benjamin Northern Power Women Case Study Transcribed by [Speaker 1] 
So it's mostly for women, where every year, they give opportunity to women to vote other women or men as well, but mostly vote women in different categories. So let's say women mentor of the year, agent of change, and all those have 13 categories. So people register, or those who already have an account into the platform can vote for the best women of the year. 
And then every first Monday of every March, there is an award show in Manchester where they give award to those that are going to win. So that's the main focus. That's the first focus of the platform. 
So award for women to celebrate gender equality, all those things. The second one is events. So they organize events basically every month. 
So there is one, okay, at least one event every month. So it can be two or more, but there is at least one event every month. And those events are basically to share. 
So they invite people that have like make it in life, like CEOs, mostly, like I said, just like CEOs, and I don't know, director and all those ladies that have even actors. And those ones will always come in those events and then give their story, encourage other women. There's also other type of event like meetup, they're going to invite some mentees, and then goes to some companies and then meets mentors there and teach them and all those things. 
So events, sorry, awards, and then events. Also, they also have podcasts, actually. Right. 
So podcast is basically almost like events, like event is basically, they invite people, right, to talk. And they also, right, and podcast is basically like, like I said before, like an actor, and it's just a podcast where they're going to talk and publish it as well on the platform. So these are the three, we also have insight, insight are basically like blog. 
But those are the three main things that we do on the platform. So award, event, and podcast. Yes. 
And then like I said, it's mostly it's about women, even if in the award, like last year, that was the first year we have one category where you can vote a male. I think it's up. Yeah, there's only one category, and we added last year, but mostly just about women. 
Yeah, because the platform itself is Northern Power Women. So that's the complete name of the platform.
[Speaker 2] 
Okay, so that sounds nice. But one thing, Ersel mentioned that when we started with them, it was one month to voting season. And it was at a critical time. 
And there were some challenges. Could you expand on that maybe? [Speaker 1] 
Yeah, when they start, they only have, they only had events, actually. And then they were only at events. And they were trying to start awards. 
That's all they had. So they only only post events, they didn't have any podcasts, or any awards. They were trying to have awards for the first time. 
But it was in the middle. That's why they gave us the project to make it a little bit better. And the platform was a bit bad, if I can put it that way, in terms of tech. 
I mean, the tech, we are using the same, but in terms of organizations, or structure, sorry, in terms of design and all those things. So we fixed everything from scratch. Actually, we start this platform from scratch, actually. 
So we like, we start everything, we start the structure of the website, the design, and we start to have awards. And actually, we build award on top of what they already started. So we fix what was there. 
We had function where in awards, you can, the judge also can just create their account, and then we can assign them judge ticket, sorry, we can assign them as judges, and then you can just go inside the platform, and then judge, and then give scores to each people. And at the end of the day, the platform calculate all those, the scores, and it's going to give us who is the best in this category, all those stuff. So all those things didn't exist before. 
So we add all of those. And we had, that's why we also had insight. And we also had podcast inside. 
But we also had things like a platform for admin didn't exist inside the platform. So we had admin, what we call admin platform where it's only access to admin where they can do all, most of the things, like assign to people badges, right, or add a new event, because even when we start the platform, if they wanted to add a new event, you had that event directly in the database. There is no place where you can type your event and add it, right? 
So we had that on the admin platform, they can just go and follow the steps of the event, images, the dates, the speakers. And then we also had things like, oh, people can actually click on a button to say that I'm going to attend to this event. And you have a
list of the people. 
And when people come to the event, you can also have the list of attendees that are on the event. So you know which people came, which people didn't come. And we also had the email systems to it. 
So when someone book, they receive an email, and then we also had the, how do you call it, schedule email. For the schedule email, we use actually Zext. So we also have the schedule email, things that, oh, okay, you registered today for an event, so they're going to remind you every seven days that you have an event, let's say, in one week and things like that, all those things. 
And we also adapt the same system of emails and stuff to nomination. Those are the things. So this platform was I'm coming. 
I'm coming. I am coming. I'm in a meeting. 
Okay. So sorry, it's my niece. All right. 
So the event was blank. That's how we had all those options. And then, yeah, that's how it is. 
We don't just have, we didn't just have like nominations and awards and podcasts. We also had chat, but they can chat actually. So you can ask a request to a mentor as a mentee. 
And then if the person accepts your request, you can start chatting. We also had forums in the platforms. Forums is not operational yet, so they're still in beta, but very soon they're going to put it public, but it's already done that they just need to test and then it's going to be public as well. 
So basically, personally, I didn't, if I have to talk about challenges, my God, personally, it will be to adapt to this platform that they gave us in the beginning. Like I said, the structure was not well-structured. Like I said, before the people that was working there, it was well-structured because maybe that's the way that we're used to. 
But as you know, get someone else's project, you need to understand how they did that. So that one was maybe the challenge that we encountered. And then actually it didn't take us much time to understand it, maybe three days to one week. 
So we knew where we were and how to continue from there. Yeah. And then the other challenges that we really find was with emails actually. 
Like to make email works was a bit, not like straight email, like scheduled email. So that was, I mean, that was, I think that was the only time me and RCL, we partnered on the project because most of the time I worked on the project alone. Only time me and RCL,
we tried to understand how we can make this work because we tried to make direct in Laravel, it didn't work. 
So we were like, what happens if we create like some sort of APIs that is going to go from Laravel and then create some sort of API and send that API to XX project and on X it's easy for us to do that. So we tried that and it works. So that was a pretty good solution we found. 
But those are the challenges that we get, I got since we started the project. I didn't, most of the time I don't get a challenge on the platform because it's, I already know what to do basically in a Laravel project. Yeah. 
[Speaker 2] 
That's, that sounds really nice. Do you have any statistics you can give me, like the number of nominations you have processed since we started? Yeah. 
[Speaker 1] 
So let me just load it in and check. So basically let's start with people. Okay. 
When we started the platform, I think we were around, we were around 3,000 people, 3,000 users. Today we are around 15,000 users and it has almost 5,000 active users per week. So 10,000 people. 
So we know that at least 5,000 people log in every week, right? It's not the same people, like different people, but 5,000 people will log every week because before that we had like 300 people only active when we get the platforms. So we try, because we also try to help, sorry, don't talk as I did. 
So anyway, we basically try to help them also in the marketing, how we can, the way we build, for example, the UI was to make people come more into the platforms, right? So all those help us to get more users and people become more active. So yeah, so we have around 15,000 registered users and we have at least 5,000 people that log in every week. 
[Speaker 2] 
Yeah. 
[Speaker 1] 
And for the award, so this year, for this year, let me start with last year. All right. So last year we had around, okay, okay, okay. 
[Speaker 2]
Yeah, yeah. 
[Speaker 1] 
Okay. So last year, we started in 2022. That was the first year that we kind of like get the platforms. 
1923, 1923, 1922 was it. Anyway, we were on the same. So every year we have an increase in nomination of 15 to 20%. 
So yeah. The first year where they did nomination without the platforms, we had more users because like Simone said, it was people didn't use the platform first, but people were using another platform where they used to to rent the place, I think. But the reason why we had more users the first year is because people wanted to know how it works, actually. 
And we didn't have any platform. So we were accepting anyone to come and vote, actually. And it was a bit messy, but they cannot make it. 
That's why they decided to start the platform to kind of like manage things a bit easier. So every year we have an increase of at least 15 to 20%. So if I calculate from last year to this year, like last year, we had around a total of 1000 nominations. 
And this year, we are already around 1007 nominations, but the nomination is still going. So yeah. So it's 11 each year. 
And every year we have at least 15 to 20% increase in nominations. Yes. So our aim is to push people because a lot of people that use the platform, they don't use it for voting. 
Most of the people using for it's like just to register to an event or to come and listen to the podcast. So we're trying to find a way with the ladies to push more people to vote. Like this year, the best way, like I can give an example, because last year we saw that most of the people don't really vote, actually. 
Like less and less people. We have to push people. So what we did this year, we allow people that don't have an account to also vote, but you have to go to the platform and then you click on vote and then you don't need to put your details. 
And this one, up to now, like this year, the people that are not having an account, we have 1931 only. So if you calculate that in 1500, 900 people that did not log in to vote. So it was a good call that I decided to have and then they approve it and then it actually pays off. 
Yeah. Because when we start nomination this year, we saw a decrease in people voting, but increasing people using the platform. So we were like, oh, why people are not using voting?
So we tried to do that. And also, with other parties on their side, because they didn't do marketing this year. So for them, they said this year is basically like a testing year, we're experimenting here. 
So we want to try to see if people can remember that we can go and vote for people without any hard anywhere. So every year what they do is, I think one month before the end of the nomination, they did a lot of ads and a lot of people come to vote. This year we didn't do anything. 
We just wanted to let things happen by itself. And then yes, we had an increase, but yeah, that's how it is. Yeah. 
So increasing nomination every year and increasing user. And we have a lot of users these days, like I think around maybe 300 per day that create their account and something like that. Yeah. 
So on user side, we are good. Now we are trying to focus more on voting. So we want to get at least 50% of increase every year instead of like 20%. 
[Speaker 2] 
So I think that's all the question I have. You answered everything. Thank you very, very much. 
[Speaker 1] 
You're welcome. If you have any questions, just send it to me. I'll be happy to answer. [Speaker 2] 
Okay. Okay. Before you go, congratulations again. 
[Speaker 1] 
Thank you so much. I'll send you some pictures. I always forget. 
[Speaker 2] 
Okay. Please do. 
[Speaker 1] 
Yeah. Thank you. Thank you so much. 
[Speaker 2] 
Thank you. Have a wonderful evening. Bye.
You too. 

"""

    result = summarizer.process_lecture(lecture_text)

    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print("\nBrief Summary:")
        print(result["brief_summary"])
        print("\nKey Points:")
        for i, point in enumerate(result["key_points"], 1):
            print(f"{i}. {point}")
        print("\nDetailed Summary:")
        print(result["detailed_summary"])