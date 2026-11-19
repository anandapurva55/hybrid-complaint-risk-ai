# Explanation review sheet (blind A/B)

For each case, read the complaint and the decision, then rate Explanation A and B from 1 (poor) to 5 (excellent) on
**faithful** (says nothing the complaint does not support), **useful** (would help a reviewer act) and **clear**, and mark
which you would rather see. The key to which is the LLM is in `explanation_review_key.json`: do not open it until you finish.

## Case 1
**Complaint:** I really liked these cords. i ordered both colors. they are very lush corduroy material and the color is very rich. since i am short, i wear them as ankle length (i am 5'1"). my only issue is that they run very large. i am normally a 25-26 and the 25 was still a bit large. unfortunately, they are all sold out of size 24 :(

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
Rated low dissatisfaction risk (high 4%, medium 18%, low 78%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

**Explanation B**  
The customer liked the cords but found them to run large.  
The models assigned a 77.5% probability to low dissatisfaction, indicating a minor issue with sizing.  
Notes: The customer is satisfied with the material and color.; The only issue is the size running large.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 2
**Complaint:** These shipped fast!!! the fit is good. these shorts have some stretch to them, which is nice. there is a green/blue color in the fine detail making them versatile with a few different colors of tops. true to size. the length is perfect for my taste. i am 5'7".

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
Rated low dissatisfaction risk (high 0%, medium 1%, low 99%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

**Explanation B**  
The customer is satisfied with the fast shipping, good fit, and versatile color of the shorts.  
The models assigned a 98.9% probability to low dissatisfaction, indicating a high level of customer satisfaction.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 3
**Complaint:** This fabric of this top is gorgeous. however, i'm 5' 5" and it hits in a really weird spot. right no the point of the hip that you don't want to accentuate. you could wear it with leggings to a casual thing, but i work in an office so its too short to wear with leggings and too long to wear with jeans. sad, i love the top 90% of it.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
Rated medium dissatisfaction risk (high 26%, medium 55%, low 19%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

**Explanation B**  
The top's length is unsuitable for the customer's work attire.  
The models assigned a 54.6% probability to medium dissatisfaction, indicating a notable but not severe issue with the product's fit.  
Notes: The customer's height and the top's length do not align well for office wear.; The customer expressed love for the top except for its length.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 4
**Complaint:** Material is so thin this shirt ripped in three places after wearing only once.

got a size larger than i needed as i have a larger chest and even though it's a bit lose the thread has ripped material on my chest and in the armpits.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 94%, medium 4%, low 1%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The shirt ripped in multiple places after a single wear due to thin material.  
The models assigned a 94.5% probability to high dissatisfaction, indicating a severe product-quality issue.  
Notes: Material is extremely thin.; Shirt ripped in three places after one wear.; Incorrect sizing despite ordering a larger size.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 5
**Complaint:** I love the cut and fit of this t. it's soft and has just enough flow to be comfortable without being unflattering. the gray floral print is very pretty. it's perfect with jeans or shorts in the summer.

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
The customer is highly satisfied with the cut, fit, softness, and appearance of the t-shirt.  
The models assigned a 99.1% probability to low dissatisfaction, indicating a positive feedback.  
Notes: 

**Explanation B**  
Rated low dissatisfaction risk (high 0%, medium 1%, low 99%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 6
**Complaint:** First of all, mine arrived in a regular denim shade, not the dark wash shown. also, the style is more like a blazer, and it was huge. i am 5'2" and ordered the petite xs but even xxs would have looked like a baggy blazer. as well, the sleeves came barely past my wrists, they are designed to cuff as the model has done? perhaps on someone taller and younger it would look edgy and chic! the quality is excellent with some nice details. this might be one to try in person, if you can.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
Rated medium dissatisfaction risk (high 34%, medium 53%, low 13%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

**Explanation B**  
The denim jacket arrived in a different color, was oversized, and the style did not match the model's depiction.  
The models assigned a 53.2% probability to medium dissatisfaction, indicating a notable but not severe issue with the product.  
Notes: Color discrepancy between product and listing.; Size did not fit as expected.; Style did not match the model's depiction.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 7
**Complaint:** Love!!!! i was super hesitant to try this dress on, it didn't look great online or on the hanger. but it looked great on! it has a very boho/70s/vintage feel. i am curvy, and it looked fabulous.

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
Rated low dissatisfaction risk (high 0%, medium 1%, low 98%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

**Explanation B**  
The dress looked great on despite initial reservations.  
The models assigned a 98.5% probability to low dissatisfaction, indicating a positive experience.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 8
**Complaint:** The blouse was not flattering on me. it would be for someone with a straight slim body type, it falls mid hips and so it accentuates my hips making them look larger. it's flared so it cannot really be tucked in.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 62%, medium 34%, low 4%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The blouse did not fit well and accentuated the wearer's hips.  
The models assigned a 61.7% probability to high dissatisfaction, indicating a significant product-quality complaint.  
Notes: The blouse did not fit as expected.; It accentuated the wearer's hips, making them look larger.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 9
**Complaint:** This shirt is thin and shapeless. the quality is h&m esque. unflattering. not a fan

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 92%, medium 6%, low 1%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The shirt is described as thin, shapeless, and unflattering, with quality comparable to H&M.  
The models assigned a 92.2% probability to high dissatisfaction, indicating a strong likelihood of customer dissatisfaction.  
Notes: The complaint mentions the shirt being thin and shapeless.; The customer compares the quality to H&M, which is generally perceived as low-end.; The customer explicitly states they are not a fan.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 10
**Complaint:** This jacket is just too small for me anyway and i am no way big. i am tall and athletic and this thing looked like it had shrunk in the wash. there are so many better fitting other types out there, this went back

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 71%, medium 26%, low 3%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The jacket appears too small and seems to have shrunk after washing.  
The models assigned a 71.2% probability to high dissatisfaction, indicating a significant product-quality issue.  
Notes: The jacket is too small for the customer.; The jacket appears to have shrunk after washing.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 11
**Complaint:** I wanted to love this shirt and thought i would based on the reviews. it runs really large however and while i love how flows it is i also looked pregnant. i am keeping though- looks nice layered under a jacket or sweater but alone i feel it looked sloppy and big....... which is sometimes just what you want so it is staying in my closet.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
Rated medium dissatisfaction risk (high 33%, medium 56%, low 11%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

**Explanation B**  
The shirt runs larger than expected and appeared oversized when worn alone.  
The models assigned a 55.9% probability to medium dissatisfaction, indicating a notable but not severe issue with the product fit.  
Notes: The customer mentioned the shirt runs large.; The customer felt the shirt looked sloppy and oversized when worn alone.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 12
**Complaint:** This dress has beautiful lines. i loved the design on the material, but the fabric looked very cheap & even was already snagged when it arrived. unfortunately, i must return it.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The dress arrived with a snag and looked cheap despite its beautiful design.  
The models assigned a 56.4% probability to high dissatisfaction, indicating a significant product-quality complaint.  
Notes: Fabric looked cheap; Arrived with a snag

**Explanation B**  
Rated high dissatisfaction risk (high 56%, medium 38%, low 6%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 13
**Complaint:** This is a lovely dress - however, there is a weird flap of fabric under the bow that doesn't sit right. it feels a little like a bib. if i could get it to sit right, i would keep it. every angle of the dress is lovely except for head on - which is a small problem.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The dress has a minor design flaw with a fabric flap under the bow.  
The models assigned a 55% probability to medium dissatisfaction, indicating a noticeable but not severe issue.  
Notes: Design flaw with fabric flap under the bow; Issue only noticeable from a head-on angle

**Explanation B**  
Rated medium dissatisfaction risk (high 15%, medium 55%, low 30%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 14
**Complaint:** Agree with the other reviewers. this top is excellent. great length, weight & bounce to it, flattering without being clingy. i'm 5'10" with a long torso, so i'm always in the market for tops with longer hemlines and decent material without being too tight or clingy. this one i love! i bought it in black, then went back for the turquoise, then two more... if that doesn't tell you what a great shirt this is, i don't know what will.

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
The customer is highly satisfied with the top, praising its length, weight, and material.  
The models assigned a 99.1% probability to low dissatisfaction, indicating strong customer satisfaction.  
Notes: 

**Explanation B**  
Rated low dissatisfaction risk (high 0%, medium 1%, low 99%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 15
**Complaint:** I wanted to like this coat so bad! i loved the bright red color and design, but when i received it, it seemed very cheap and just too large. i returned it.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The coat appeared cheap and was too large, leading to a return.  
The models assigned a 62.5% probability to high dissatisfaction, indicating a significant product-quality complaint.  
Notes: The customer found the coat to be cheap and too large.; The customer returned the coat.

**Explanation B**  
Rated high dissatisfaction risk (high 62%, medium 33%, low 4%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 16
**Complaint:** I looked forward to this dress.  it's a lovely design.  it's a heavy material that doesn't have the most flattering fit.  the green lace has snags along the hemlines which is a disappointment.  i'm returning the dress.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The dress has a lovely design but is not flattering and has snags on the lace.  
The models assigned a 55.4% probability to medium dissatisfaction, indicating a notable but not severe issue.  
Notes: The dress has snags on the lace.; The material is heavy and not flattering.

**Explanation B**  
Rated medium dissatisfaction risk (high 31%, medium 55%, low 14%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 17
**Complaint:** Love the wine colored cords-- but i tried them and i was disappointed because my normal size was too small. i usually have to size down in pilcro pants but not these. (i'm an 8, and the 30 wasn't going to button!) i love the slight cropped flare! and the color! did i mention the color? but vanity wouldn't allow me to buy them. hopefully i just got a mislabeled pair!

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The customer was disappointed that their normal size was too small for the cords.  
The models assigned a 24.1% probability to medium dissatisfaction, indicating a sizing issue.  
Notes: The customer's normal size was too small.; The customer had to size down for other products but not for these cords.

**Explanation B**  
Rated medium dissatisfaction risk (high 4%, medium 24%, low 72%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 18
**Complaint:** I really wanted to like this shirt, but it made me look 9 months pregnant. which would be fine if i were pregnant.  it sits wired in the bust and flows out in the front and back. not flattering at all.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The shirt is unflattering and does not fit well, making the wearer look pregnant.  
The models assigned a 90.8% probability to high dissatisfaction due to the poor fit and unflattering appearance of the shirt.  
Notes: The shirt does not fit well and is unflattering.; The customer feels the shirt makes them look pregnant.

**Explanation B**  
Rated high dissatisfaction risk (high 91%, medium 8%, low 1%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie
