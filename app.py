           # app.py
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import nltk
import os
nltk.download('punkt')
from newspaper import Article
import groq
from azure.cognitiveservices.search.websearch import WebSearchClient
from msrest.authentication import CognitiveServicesCredentials
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_migrate import Migrate

app = Flask(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fact_checker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
#db.create_all()
# Initialize the database
# After initializing `db`
migrate = Migrate(app, db)


class Claim(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    date_checked = db.Column(db.DateTime, default=datetime.utcnow)
    veracity_probability = db.Column(db.Float)
    veracity_justification = db.Column(db.Text)
    final_truth_score = db.Column(db.Float)
    overall_scores = db.relationship('OverallCRAAPScore', backref='claim', lazy=True)
    sources = db.relationship('Source', backref='claim', lazy=True)


class Source(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    claim_id = db.Column(db.Integer, db.ForeignKey('claim.id'), nullable=False)
    name = db.Column(db.String(512), nullable=False)
    url = db.Column(db.String(512), nullable=False)
    snippet = db.Column(db.Text)
    date_last_crawled = db.Column(db.String(128))
    intent_category = db.Column(db.String(64))
    intent_explanation = db.Column(db.Text)
    craap_scores = db.relationship('CRAAPScore', backref='source', lazy=True)


class CRAAPScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('source.id'), nullable=False)
    criterion = db.Column(db.String(64), nullable=False)
    score = db.Column(db.Float, nullable=False)
    explanation = db.Column(db.Text)

class OverallCRAAPScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    claim_id = db.Column(db.Integer, db.ForeignKey('claim.id'), nullable=False)
    criterion = db.Column(db.String(64), nullable=False)
    score = db.Column(db.Float, nullable=False)

def calculate_final_truth_score(overall_craap_scores, veracity_probability):
    # Normalize CRAAP score (assuming maximum CRAAP score per criterion is 10)
    normalized_craap_score = sum(overall_craap_scores.values()) / (10 * len(overall_craap_scores))
    # Assign weights
    craap_weight = 0.5
    veracity_weight = 0.5
    # Calculate final score
    final_score = (normalized_craap_score * craap_weight) + (veracity_probability * veracity_weight)
    return final_score


# Initialize Groq client
client = groq.Groq()


def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return ''

def extract_claims(text):
    prompt = f"""
    Analyze the following text and extract the main factual claims (up to 5). Provide each claim in a numbered list.

    Text: {text}

    Format:
    1. [First claim]
    2. [Second claim]
    3. [Third claim]
    """
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts factual claims from text."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2
    )
    claims_text = response.choices[0].message.content
    # Extract claims from the response
    claims = []
    for line in claims_text.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() and '.' in line):
            claim = line.split('.', 1)[1].strip()
            claims.append(claim)
    return claims


def search_sources_for_claim(claim):
    import os
    import requests
    import html
    from bs4 import BeautifulSoup

    def clean_text(text):
        text = html.unescape(text)
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    try:
        # Get the Bing Search API key from environment variable
        subscription_key = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')
        if not subscription_key:
            raise ValueError("Bing Search API key not found. Set the 'BING_SEARCH_V7_SUBSCRIPTION_KEY' environment variable.")

        search_url = "https://api.bing.microsoft.com/v7.0/search"

        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        params = {"q": claim, "textDecorations": True, "textFormat": "HTML"}

        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        sources = []
        if "webPages" in search_results and "value" in search_results["webPages"]:
            for result in search_results["webPages"]["value"][:5]:  # Limit to top 5 results
                name = clean_text(result.get('name', ''))
                snippet = clean_text(result.get('snippet', ''))
                source = {
                    'name': name,
                    'url': result.get('url'),
                    'snippet': snippet,
                    'date_last_crawled': result.get('dateLastCrawled')
                }
                sources.append(source)
        else:
            print(f"No web pages found for claim: '{claim}'")
        return sources
    except Exception as e:
        print(f"Error searching for claim '{claim}': {e}")
        return []



def compute_craap_score(claim, source):
    prompt = f"""
    Evaluate the following source for the claim "{claim}" using the CRAAP test.

    Source:
    Title: {source['name']}
    URL: {source['url']}
    Snippet: {source['snippet']}
    Date Last Crawled: {source['date_last_crawled']}

    Assess each of the following criteria on a scale from 0 (lowest) to 10 (highest):
    - Currency (Is the information up-to-date?)
    - Relevance (Does the source relate to the claim?)
    - Authority (Is the author/publisher/source reputable?)
    - Accuracy (Is the information reliable, truthful, and correct?)
    - Purpose (Is the purpose of the information clear? Is it free of bias?)

    Provide a score for each criterion and a brief explanation.

    Format:
    Currency: [Score] - [Explanation]
    Relevance: [Score] - [Explanation]
    Authority: [Score] - [Explanation]
    Accuracy: [Score] - [Explanation]
    Purpose: [Score] - [Explanation]
    """

    response = client.chat.completions.create(
        model="llama3-groq-70b-8192-tool-use-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that evaluates sources based on the CRAAP test."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2
    )

    content = response.choices[0].message.content

    # Parse the response to extract scores
    craap_scores = {}
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('Currency:'):
            score_explanation = line.split(':',1)[1].strip()
            score, explanation = score_explanation.split('-',1)
            craap_scores['Currency'] = {'score': float(score.strip()), 'explanation': explanation.strip()}
        elif line.startswith('Relevance:'):
            score_explanation = line.split(':',1)[1].strip()
            score, explanation = score_explanation.split('-',1)
            craap_scores['Relevance'] = {'score': float(score.strip()), 'explanation': explanation.strip()}
        elif line.startswith('Authority:'):
            score_explanation = line.split(':',1)[1].strip()
            score, explanation = score_explanation.split('-',1)
            craap_scores['Authority'] = {'score': float(score.strip()), 'explanation': explanation.strip()}
        elif line.startswith('Accuracy:'):
            score_explanation = line.split(':',1)[1].strip()
            score, explanation = score_explanation.split('-',1)
            craap_scores['Accuracy'] = {'score': float(score.strip()), 'explanation': explanation.strip()}
        elif line.startswith('Purpose:'):
            score_explanation = line.split(':',1)[1].strip()
            score, explanation = score_explanation.split('-',1)
            craap_scores['Purpose'] = {'score': float(score.strip()), 'explanation': explanation.strip()}
    return craap_scores


def compute_overall_craap_score(craap_scores_list):
    # craap_scores_list is a list of dictionaries of CRAAP scores for each source
    aggregated_scores = {'Currency': 0, 'Relevance': 0, 'Authority': 0, 'Accuracy': 0, 'Purpose': 0}
    num_sources = len(craap_scores_list)
    for scores in craap_scores_list:
        for criterion in aggregated_scores.keys():
            aggregated_scores[criterion] += scores[criterion]['score']
    # Compute average scores
    for criterion in aggregated_scores.keys():
        aggregated_scores[criterion] /= num_sources
    return aggregated_scores

  

def extract_and_verify_claims(text):
    claims = extract_claims(text)
    results = []
    
    for claim_text in claims:
        # Initialize variables
        overall_craap_scores = None
        veracity_assessment = None
        final_truth_score = None

        print(f"Processing claim: '{claim_text}'")

        # Check if the claim already exists in the database
        claim = Claim.query.filter_by(text=claim_text).first()
        if not claim:
            claim = Claim(text=claim_text)
            db.session.add(claim)
            db.session.commit()

        sources = search_sources_for_claim(claim_text)
        if not sources:
            print(f"No sources found for claim: '{claim_text}'")
            craap_scores_list = []
            sources_data = []
        else:
            craap_scores_list = []
            sources_data = []
            for source_data in sources:
                # Save source to database
                source = Source(
                    claim_id=claim.id,
                    name=source_data['name'],
                    url=source_data['url'],
                    snippet=source_data['snippet'],
                    date_last_crawled=source_data['date_last_crawled']
                )
                # Intentionality Categorization
                intent_categorization = categorize_source_intent(source_data)
                source.intent_category = intent_categorization['category']
                source.intent_explanation = intent_categorization['explanation']
                db.session.add(source)
                db.session.commit()

                # Prepare CRAAP scores
                craap_scores = compute_craap_score(claim_text, source_data)
                # Append the CRAAP scores dictionary to the list
                craap_scores_list.append(craap_scores)

                # Prepare source-specific CRAAP scores for display
                source_craap_scores_list = [
                    {
                        'criterion': criterion,
                        'score': details['score'],
                        'explanation': details['explanation']
                    }
                    for criterion, details in craap_scores.items()
                    if criterion != 'source'
                ]

                # Add source data to sources_data list
                sources_data.append({
                    'name': source.name,
                    'url': source.url,
                    'snippet': source.snippet,
                    'intent_category': source.intent_category,
                    'intent_explanation': source.intent_explanation,
                    'craap_scores': source_craap_scores_list
                })

            # Compute overall CRAAP scores
            overall_craap_scores = compute_overall_craap_score(craap_scores_list)
            # Assess the claim's veracity
            veracity_assessment = assess_claim_veracity(claim_text, sources)

        # Calculate final truth score
        if overall_craap_scores and veracity_assessment:
            final_truth_score = calculate_final_truth_score(
                overall_craap_scores,
                veracity_assessment['probability']
            )
            # Update claim with veracity assessment and final truth score
            claim.veracity_probability = veracity_assessment['probability']
            claim.veracity_justification = veracity_assessment['justification']
            claim.final_truth_score = final_truth_score
            db.session.commit()
        else:
            final_truth_score = None

        # Add everything to the results list
        results.append({
            'claim': claim_text,
            'sources': sources_data if sources else [],
            'overall_craap_scores': overall_craap_scores,
            'veracity_assessment': veracity_assessment,
            'final_truth_score': final_truth_score
        })
    return results



def assess_claim_veracity(claim, sources):
    try:
        # Combine the content from the top sources
        source_texts = ''
        for source in sources:
            source_texts += f"- {source['snippet']}\n"

        # Prepare the prompt
        prompt = f"""
You are an expert fact-checker. Given the following claim and evidence from sources, assess the truthfulness of the claim. Provide a probability score between 0 and 1, where 1 indicates the claim is definitely true, and 0 indicates it is definitely false. Also, provide a brief justification.

Claim:
"{claim}"

Evidence:
{source_texts}

Format:
Probability: [value between 0 and 1]
Justification: [Your brief justification]
"""

        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that assesses the truthfulness of claims based on evidence."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )

        # Parse the response
        content = response.choices[0].message.content.strip()
        probability = None
        justification = ''
        lines = content.split('\n')
        for line in lines:
            if line.startswith('Probability:'):
                probability = float(line.split('Probability:')[1].strip())
            elif line.startswith('Justification:'):
                justification = line.split('Justification:')[1].strip()

        return {
            'probability': probability,
            'justification': justification
        }
    except Exception as e:
        print(f"Error assessing claim veracity: {e}")
        return None


def categorize_source_intent(source):
    try:
        # Prepare the prompt
        prompt = f"""
You are an expert analyst. Given the following source content, categorize the source's intent into one of the following categories:

1. News/Journalism
2. Opinion/Editorial
3. Scientific/Scholarly
4. Marketing/Advertising
5. Entertainment
6. Propaganda
7. Satire/Parody
8. Advocacy/Non-Profit
9. Personal Blog/Opinion
10. Government/Official
11. Educational
12. Social Media Post

Source Snippet:
"{source['snippet']}"

Format:
Category Number: [Select the most appropriate category number]
Explanation: [Briefly explain why this category was chosen]
"""

        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that categorizes sources based on their intent."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.2
        )

        # Parse the response
        content = response.choices[0].message.content.strip()
        category_number = None
        explanation = ''
        lines = content.split('\n')
        for line in lines:
            if line.startswith('Category Number:'):
                category_number = int(line.split('Category Number:')[1].strip())
            elif line.startswith('Explanation:'):
                explanation = line.split('Explanation:')[1].strip()

        # Map the category number to the category name
        categories = {
            1: 'News/Journalism',
            2: 'Opinion/Editorial',
            3: 'Scientific/Scholarly',
            4: 'Marketing/Advertising',
            5: 'Entertainment',
            6: 'Propaganda',
            7: 'Satire/Parody',
            8: 'Advocacy/Non-Profit',
            9: 'Personal Blog/Opinion',
            10: 'Government/Official',
            11: 'Educational',
            12: 'Social Media Post'
        }
        category_name = categories.get(category_number, 'Unknown')

        return {
            'category': category_name,
            'explanation': explanation
        }
    except Exception as e:
        print(f"Error categorizing source intent: {e}")
        return {
            'category': 'Unknown',
            'explanation': 'Could not determine the category.'
        }




def interpret_probability(probability):
    if probability is None:
        return 'Unknown'
    elif probability >= 0.7:
        return 'True'
    elif probability <= 0.3:
        return 'False'
    else:
        return 'Uncertain'


def interpret_final_score(final_score):
    if final_score is None:
        return 'Unknown'
    elif final_score >= 0.7:
        return 'True'
    elif final_score <= 0.3:
        return 'False'
    else:
        return 'Uncertain'




@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_type = request.form['input_type']
        content = request.form['content']

        if input_type == 'url':
            text = extract_text_from_url(content)
        else:
            text = content

        fact_check_results = extract_and_verify_claims(text)

        return render_template('results.html', results=fact_check_results)
    return render_template('index.html')

@app.route('/news')
def news():
    claims = Claim.query.order_by(Claim.date_checked.desc()).all()
    claims_data = []
    for claim in claims:
        overall_scores = {score.criterion: score.score for score in claim.overall_scores}
        sources = [{
            'name': source.name,
            'url': source.url,
            'intent_category': source.intent_category
        } for source in claim.sources]
        claims_data.append({
            'id': claim.id,
            'text': claim.text,
            'date_checked': claim.date_checked,
            'overall_scores': overall_scores,
            'veracity_probability': claim.veracity_probability,
            'veracity_justification': claim.veracity_justification,
            'final_truth_score': claim.final_truth_score,
            'sources': sources
        })
    return render_template('news.html', claims=claims_data)

@app.route('/claim/<int:claim_id>')
def claim_detail(claim_id):
    claim = Claim.query.get_or_404(claim_id)
    overall_scores = {score.criterion: score.score for score in claim.overall_scores}
    sources_data = []
    for source in claim.sources:
        craap_scores = {}
        for score in source.craap_scores:
            craap_scores[score.criterion] = {
                'score': score.score,
                'explanation': score.explanation
            }
        sources_data.append({
            'name': source.name,
            'url': source.url,
            'snippet': source.snippet,
            'intent_categorization': {
                'category': source.intent_category,
                'explanation': source.intent_explanation
            },
            'craap_scores': craap_scores
        })
    veracity_assessment = {
        'probability': claim.veracity_probability,
        'justification': claim.veracity_justification
    }
    final_truth_score = claim.final_truth_score
    return render_template(
        'claim_detail.html',
        claim=claim,
        overall_scores=overall_scores,
        sources=sources_data,
        veracity_assessment=veracity_assessment,
        final_truth_score=final_truth_score
    )




@app.route('/about')
def about():
    current_year = datetime.now().year
    return render_template('about.html', current_year=current_year)

@app.context_processor
def utility_functions():
    def interpret_probability(probability):
        if probability is None:
            return 'Unknown'
        elif probability >= 0.7:
            return 'True'
        elif probability <= 0.3:
            return 'False'
        else:
            return 'Uncertain'

    def interpret_final_score(final_score):
        if final_score is None:
            return 'Unknown'
        elif final_score >= 0.7:
            return 'True'
        elif final_score <= 0.3:
            return 'False'
        else:
            return 'Uncertain'

    return dict(
        interpret_probability=interpret_probability,
        interpret_final_score=interpret_final_score
    )



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8080)