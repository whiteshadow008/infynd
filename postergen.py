"""
AI Social Media Poster Generator - Ultra Realistic with 3D Elements
Generates high-quality, realistic social media posters with advanced visual effects,
3D elements, shadows, gradients, and professional design elements
"""

import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
from datetime import datetime
import math

class SocialMediaPosterGenerator:
    def __init__(self):
        self.platform_sizes = {
            'Instagram': (1080, 1080),
            'Instagram Story': (1080, 1920),
            'Facebook': (1200, 630),
            'Twitter/X': (1200, 675),
            'LinkedIn': (1200, 627)
        }
        
        self.color_schemes = {
            'Technology': {
                'primary': '#0a0e27',
                'secondary': '#1a1f3a',
                'accent': '#6c5ce7',
                'highlight': '#00d4ff',
                'gradient_start': '#667eea',
                'gradient_end': '#764ba2',
                'text': '#ffffff',
                'shadow': '#000033'
            },
            'Healthcare': {
                'primary': '#e8f8f5',
                'secondary': '#a8e6cf',
                'accent': '#56ab91',
                'highlight': '#81c784',
                'gradient_start': '#4facfe',
                'gradient_end': '#00f2fe',
                'text': '#1b5e20',
                'shadow': '#2d5f3f'
            },
            'Finance': {
                'primary': '#0a2342',
                'secondary': '#1c3d5a',
                'accent': '#2ca58d',
                'highlight': '#84bc9c',
                'gradient_start': '#134e5e',
                'gradient_end': '#71b280',
                'text': '#ffffff',
                'shadow': '#000a1f'
            },
            'Retail': {
                'primary': '#ff6b6b',
                'secondary': '#ff8787',
                'accent': '#4ecdc4',
                'highlight': '#ffe66d',
                'gradient_start': '#fa709a',
                'gradient_end': '#fee140',
                'text': '#2d3436',
                'shadow': '#8b0000'
            },
            'Food & Beverage': {
                'primary': '#fff3e0',
                'secondary': '#ffe0b2',
                'accent': '#ff6f00',
                'highlight': '#ffab00',
                'gradient_start': '#f83600',
                'gradient_end': '#fe8c00',
                'text': '#bf360c',
                'shadow': '#e65100'
            },
            'Education': {
                'primary': '#283593',
                'secondary': '#3949ab',
                'accent': '#5c6bc0',
                'highlight': '#9fa8da',
                'gradient_start': '#667eea',
                'gradient_end': '#764ba2',
                'text': '#ffffff',
                'shadow': '#1a237e'
            },
            'Real Estate': {
                'primary': '#263238',
                'secondary': '#37474f',
                'accent': '#546e7a',
                'highlight': '#90a4ae',
                'gradient_start': '#232526',
                'gradient_end': '#414345',
                'text': '#ffffff',
                'shadow': '#000000'
            },
            'Fashion': {
                'primary': '#1a1a1a',
                'secondary': '#2d2d2d',
                'accent': '#d4af37',
                'highlight': '#ffffff',
                'gradient_start': '#434343',
                'gradient_end': '#000000',
                'text': '#ffffff',
                'shadow': '#000000'
            },
            'Fitness': {
                'primary': '#ff5722',
                'secondary': '#ff7043',
                'accent': '#ff9800',
                'highlight': '#ffc107',
                'gradient_start': '#f83600',
                'gradient_end': '#f9d423',
                'text': '#ffffff',
                'shadow': '#bf360c'
            },
            'Travel': {
                'primary': '#006064',
                'secondary': '#00838f',
                'accent': '#0288d1',
                'highlight': '#4fc3f7',
                'gradient_start': '#4facfe',
                'gradient_end': '#00f2fe',
                'text': '#ffffff',
                'shadow': '#004d40'
            }
        }
    
    def generate_description(self, company, industry, topic, tone):
        """Generate comprehensive AI-powered description with emojis and engagement hooks"""
        
        tone_templates = {
            'Professional': [
                f"🚀 {company} is revolutionizing {industry.lower()} through {topic}.\n\n"
                f"Our innovative approach combines cutting-edge technology with deep industry expertise to deliver exceptional results. "
                f"We're committed to excellence, sustainability, and creating lasting value for our clients.\n\n"
                f"✨ Key Highlights:\n"
                f"• Industry-leading solutions\n"
                f"• Proven track record of success\n"
                f"• Customer-centric approach\n"
                f"• Future-ready strategies\n\n"
                f"Partner with us to transform your business. Let's shape the future together! 💼",
                
                f"🎯 Introducing {company}'s game-changing approach to {topic}.\n\n"
                f"In today's rapidly evolving {industry.lower()} landscape, staying ahead means embracing innovation. "
                f"Our team of experts brings decades of combined experience to help you achieve your goals.\n\n"
                f"📊 What sets us apart:\n"
                f"• Data-driven insights\n"
                f"• Customized solutions\n"
                f"• Measurable results\n"
                f"• 24/7 dedicated support\n\n"
                f"Ready to elevate your business? Connect with us today! 🤝",
                
                f"💡 {company} presents: The future of {topic} in {industry.lower()}.\n\n"
                f"We're not just keeping pace with industry changes – we're setting the standard. "
                f"Our comprehensive solutions are designed to address your unique challenges and unlock new opportunities.\n\n"
                f"🌟 Our Promise:\n"
                f"• Innovation-driven solutions\n"
                f"• Transparent communication\n"
                f"• Scalable growth strategies\n"
                f"• Proven ROI\n\n"
                f"Join the leaders who trust {company}. Let's build success together! 📈"
            ],
            
            'Casual & Friendly': [
                f"Hey there! 👋 It's {company}, and we're SO excited to share something awesome with you!\n\n"
                f"We've been working on {topic} and honestly? The results are mind-blowing! 🤯 "
                f"Our team has poured their hearts into making {industry.lower()} more accessible, fun, and impactful.\n\n"
                f"Here's what's cool:\n"
                f"✅ Super easy to get started\n"
                f"✅ Real results, real fast\n"
                f"✅ Amazing community support\n"
                f"✅ We're here for you every step\n\n"
                f"Wanna join the journey? Slide into our DMs! We'd love to chat! 💬💙",
                
                f"Guess what?! 🎉 {company} just leveled up!\n\n"
                f"We're bringing you the freshest take on {topic} in the {industry.lower()} world. "
                f"Think of us as your favorite coffee shop – warm, welcoming, and always serving up something special! ☕\n\n"
                f"Why people love us:\n"
                f"❤️ We keep it real and honest\n"
                f"❤️ No complicated jargon\n"
                f"❤️ Fast, friendly service\n"
                f"❤️ We celebrate your wins!\n\n"
                f"Come hang with us! Your success story starts here! 🌟",
                
                f"Hey friends! 🌈 {company} here with some seriously cool news!\n\n"
                f"We're all about making {topic} work for YOU. No fluff, no fuss – just real solutions that actually make a difference. "
                f"Our vibe? Professional results with a personal touch! 💪\n\n"
                f"What makes us different:\n"
                f"🎨 Creative solutions\n"
                f"🤗 Super friendly team\n"
                f"⚡ Quick responses\n"
                f"🎯 Results that matter\n\n"
                f"Let's connect and make magic happen! Drop us a message! ✉️✨"
            ],
            
            'Inspiring': [
                f"✨ Every great achievement starts with a bold vision. ✨\n\n"
                f"{company} believes in the transformative power of {topic}. "
                f"We're not just building solutions – we're creating movements that shape the future of {industry.lower()}.\n\n"
                f"🌟 Your journey to excellence includes:\n"
                f"• Unlimited potential\n"
                f"• Fearless innovation\n"
                f"• Meaningful impact\n"
                f"• Lasting legacy\n\n"
                f"Dream bigger. Reach higher. Together, we're unstoppable! 🚀\n\n"
                f"The future isn't something that happens – it's something we CREATE. Are you ready? 💫",
                
                f"🌅 Success isn't a destination. It's a journey of continuous growth.\n\n"
                f"At {company}, we're inspired every single day by {topic} and its power to transform lives in {industry.lower()}. "
                f"Your dreams aren't just valid – they're essential. And we're here to help you achieve them.\n\n"
                f"💎 Believe in:\n"
                f"• Your unique vision\n"
                f"• The power of persistence\n"
                f"• Innovation as a mindset\n"
                f"• Impact over perfection\n\n"
                f"The world needs what you have to offer. Let's make it happen! 🌟\n\n"
                f"Remember: Every expert was once a beginner who refused to give up. Your time is NOW! ⚡",
                
                f"🎯 Champions aren't born. They're made through dedication, vision, and the courage to begin.\n\n"
                f"{company} is your partner in this incredible journey. Through {topic}, we're reimagining what's possible in {industry.lower()}. "
                f"Your potential is limitless, and your impact will be profound.\n\n"
                f"🏆 Embrace:\n"
                f"• Challenges as opportunities\n"
                f"• Growth as a lifestyle\n"
                f"• Innovation as courage\n"
                f"• Success as a team effort\n\n"
                f"Rise up. Stand out. Make your mark! 🌠\n\n"
                f"The future belongs to those who believe in the beauty of their dreams. Let's build yours! 💪✨"
            ],
            
            'Humorous': [
                f"🎭 Plot twist: {company} actually makes {topic} FUN! (We know, crazy right?)\n\n"
                f"Who knew {industry.lower()} could be this entertaining? Spoiler alert: We did! 😎\n\n"
                f"Here's the deal:\n"
                f"☕ We're probably running on coffee\n"
                f"🎉 But we're DEFINITELY delivering results\n"
                f"🤓 We speak human (not robot)\n"
                f"😄 We actually enjoy what we do!\n\n"
                f"Join the fun side of business. We have cookies! 🍪\n\n"
                f"P.S. - Your competitors are probably taking themselves too seriously. Don't be like them! 😉",
                
                f"🚨 BREAKING NEWS: {company} has officially made {topic} cool! 🎉\n\n"
                f"Scientists are baffled. Competitors are confused. Customers are THRILLED! "
                f"Welcome to {industry.lower()} reimagined (with 100% more awesome and 0% boring). 🌟\n\n"
                f"What you get:\n"
                f"✨ Magic (results, actually)\n"
                f"🎪 Fun times\n"
                f"💪 Real solutions\n"
                f"🤝 Good vibes only\n\n"
                f"Why so serious? Let's make business fun again! 🎊\n\n"
                f"Warning: Side effects may include success, happiness, and telling all your friends about us! 😁",
                
                f"📢 Attention humans! {company} has an important announcement...\n\n"
                f"We've cracked the code on {topic}! Turns out, {industry.lower()} doesn't have to be boring. "
                f"Who knew?! (Okay, we did, but we're humble... mostly) 😄\n\n"
                f"The secret sauce:\n"
                f"🎨 Creativity + Logic = WIN\n"
                f"🍕 Pizza Fridays (yes, really)\n"
                f"💡 Bright ideas (and bright people)\n"
                f"🎯 Hitting goals, not walls\n\n"
                f"Ready to join the party? Bring your sense of humor! 🎉\n\n"
                f"Remember: Life's too short for boring business! Let's have some fun! 🚀😎"
            ],
            
            'Urgent': [
                f"⚡ URGENT: Don't miss this game-changing opportunity! ⚡\n\n"
                f"{company} is revolutionizing {topic} in {industry.lower()}, and time is of the essence! "
                f"Early adopters are already seeing incredible results, and spots are filling FAST.\n\n"
                f"🔥 Act NOW to secure:\n"
                f"• Priority access\n"
                f"• Exclusive benefits\n"
                f"• Competitive advantage\n"
                f"• Limited-time offer\n\n"
                f"⏰ The clock is ticking! Your competitors won't wait, and neither should you!\n\n"
                f"Contact us TODAY before it's too late! This opportunity won't last! 🚨",
                
                f"🚨 ATTENTION {industry} PROFESSIONALS! 🚨\n\n"
                f"The landscape is changing FAST, and {company} has the solution you've been searching for! "
                f"{topic} is no longer optional – it's ESSENTIAL for survival in today's market.\n\n"
                f"⚡ Critical Action Items:\n"
                f"• Assess your current position\n"
                f"• Identify growth opportunities\n"
                f"• Implement NOW, not later\n"
                f"• Secure your competitive edge\n\n"
                f"⏳ Time-sensitive opportunity ends soon!\n\n"
                f"Don't be left behind while others surge ahead! Reach out IMMEDIATELY! 📞💼",
                
                f"🔴 BREAKING: Major shifts happening in {industry.lower()} RIGHT NOW! 🔴\n\n"
                f"{company} announces groundbreaking advancements in {topic}! "
                f"This is what industry leaders have been waiting for – and it's available for a LIMITED TIME ONLY!\n\n"
                f"⚡ Why you can't afford to wait:\n"
                f"• Market conditions are optimal NOW\n"
                f"• Competition is intensifying DAILY\n"
                f"• Opportunities are closing FAST\n"
                f"• Success requires IMMEDIATE action\n\n"
                f"🔥 Your competitors are already moving! Will you lead or follow?\n\n"
                f"Contact us THIS INSTANT! Tomorrow might be too late! ⏰🚀"
            ]
        }
        
        templates = tone_templates.get(tone, tone_templates['Professional'])
        return random.choice(templates)
    
    def generate_hashtags(self, company, industry, topic, platform):
        """Generate comprehensive, strategic hashtags"""
        
        company_tag = company.replace(' ', '').replace('&', 'And')
        topic_words = topic.split()
        topic_tag = ''.join(word.capitalize() for word in topic_words)
        industry_tag = industry.replace(' & ', '').replace(' ', '')
        
        hashtag_counts = {
            'Instagram': 30,
            'Instagram Story': 10,
            'Facebook': 5,
            'Twitter/X': 3,
            'LinkedIn': 5
        }
        
        # Comprehensive hashtag pool
        hashtags = [
            f"#{company_tag}",
            f"#{topic_tag}",
            f"#{industry_tag}",
            f"#{industry_tag}Industry",
            f"#{industry_tag}Trends",
            "#Business",
            "#BusinessGrowth",
            "#Innovation",
            "#InnovationHub",
            "#Growth",
            "#Success",
            "#SuccessStory",
            "#Entrepreneurship",
            "#Entrepreneur",
            "#Marketing",
            "#DigitalMarketing",
            "#SocialMedia",
            "#SocialMediaMarketing",
            "#ContentMarketing",
            "#BrandBuilding",
            "#Branding",
            "#Leadership",
            "#BusinessLeadership",
            "#Strategy",
            "#BusinessStrategy",
            "#Technology",
            "#TechInnovation",
            "#Future",
            "#FutureTech",
            "#Trending",
            "#TrendingNow",
            "#Inspiration",
            "#Motivation",
            "#MotivationMonday",
            f"#{platform.replace('/', '').replace(' ', '')}",
            "#ContentCreator",
            "#SmallBusiness",
            "#SmallBusinessOwner",
            "#Startup",
            "#StartupLife",
            "#Professional",
            "#CareerGrowth",
            "#Networking",
            "#BusinessTips",
            "#SuccessMindset"
        ]
        
        count = hashtag_counts.get(platform, 10)
        selected = hashtags[:count]
        
        return ' '.join(selected)
    
    def create_3d_sphere(self, size, color, position, alpha=255):
        """Create a 3D sphere with gradient shading"""
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        center = size // 2
        for r in range(size//2, 0, -1):
            # Calculate light gradient
            brightness = int((r / (size//2)) * 255)
            r_val = min(255, color[0] + brightness//3)
            g_val = min(255, color[1] + brightness//3)
            b_val = min(255, color[2] + brightness//3)
            
            current_alpha = int(alpha * (r / (size//2)))
            draw.ellipse(
                [center-r, center-r, center+r, center+r],
                fill=(r_val, g_val, b_val, current_alpha)
            )
        
        return img
    
    def create_3d_cube(self, size, color):
        """Create a 3D isometric cube"""
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Define cube vertices
        w, h = size//2, size//2
        
        # Top face
        top_face = [
            (w, h//4),
            (w*1.5, h//2),
            (w, h*0.75),
            (w*0.5, h//2)
        ]
        
        # Left face
        left_face = [
            (w*0.5, h//2),
            (w, h*0.75),
            (w, h*1.25),
            (w*0.5, h)
        ]
        
        # Right face
        right_face = [
            (w, h*0.75),
            (w*1.5, h//2),
            (w*1.5, h),
            (w, h*1.25)
        ]
        
        # Draw faces with different shades
        draw.polygon(left_face, fill=(*self.darken_color(color, 0.6), 200))
        draw.polygon(right_face, fill=(*self.darken_color(color, 0.8), 200))
        draw.polygon(top_face, fill=(*color, 220))
        
        return img
    
    def darken_color(self, color, factor):
        """Darken a color by a factor"""
        return tuple(int(c * factor) for c in color)
    
    def create_gradient(self, size, color1, color2, direction='vertical'):
        """Create smooth gradient"""
        img = Image.new('RGB', size, color1)
        draw = ImageDraw.Draw(img)
        
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        if direction == 'vertical':
            for y in range(size[1]):
                ratio = y / size[1]
                r = int(r1 + (r2 - r1) * ratio)
                g = int(g1 + (g2 - g1) * ratio)
                b = int(b1 + (b2 - b1) * ratio)
                draw.line([(0, y), (size[0], y)], fill=(r, g, b))
        else:  # horizontal
            for x in range(size[0]):
                ratio = x / size[0]
                r = int(r1 + (r2 - r1) * ratio)
                g = int(g1 + (g2 - g1) * ratio)
                b = int(b1 + (b2 - b1) * ratio)
                draw.line([(x, 0), (x, size[1])], fill=(r, g, b))
        
        return img
    
    def create_radial_gradient(self, size, center_color, edge_color):
        """Create radial gradient"""
        img = Image.new('RGB', size, edge_color)
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        r1, g1, b1 = center_color
        r2, g2, b2 = edge_color
        
        for y in range(size[1]):
            for x in range(size[0]):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                ratio = min(distance / max_distance, 1.0)
                
                r = int(r1 + (r2 - r1) * ratio)
                g = int(g1 + (g2 - g1) * ratio)
                b = int(b1 + (b2 - b1) * ratio)
                
                draw.point((x, y), fill=(r, g, b))
        
        return img
    
    def create_modern_3d_design(self, size, company, industry, topic, colors):
        """Create modern design with 3D elements and visual effects"""
        
        # Create base gradient
        gradient_start = self.hex_to_rgb(colors['gradient_start'])
        gradient_end = self.hex_to_rgb(colors['gradient_end'])
        img = self.create_gradient(size, gradient_start, gradient_end, 'vertical')
        
        # Create overlay layer
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        
        # Add 3D spheres
        sphere_color = self.hex_to_rgb(colors['accent'])
        sphere_positions = [
            (size[0]//4, size[1]//6, 200, 150),
            (size[0]*3//4, size[1]//3, 250, 120),
            (size[0]//2, size[1]*5//6, 180, 100)
        ]
        
        for x, y, sphere_size, alpha in sphere_positions:
            sphere = self.create_3d_sphere(sphere_size, sphere_color, (x, y), alpha)
            overlay.paste(sphere, (x - sphere_size//2, y - sphere_size//2), sphere)
        
        # Add 3D cubes
        cube_color = self.hex_to_rgb(colors['highlight'])
        cube_positions = [
            (size[0]//6, size[1]//2, 150),
            (size[0]*5//6, size[1]*2//3, 120)
        ]
        
        for x, y, cube_size in cube_positions:
            cube = self.create_3d_cube(cube_size, cube_color)
            overlay.paste(cube, (x - cube_size//2, y - cube_size//2), cube)
        
        # Add abstract geometric patterns
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Flowing curves
        for i in range(5):
            y_offset = size[1] // 6 * (i + 1)
            points = []
            for x in range(0, size[0] + 100, 50):
                y = y_offset + int(30 * math.sin(x / 100 + i))
                points.append((x, y))
            
            if len(points) > 1:
                overlay_draw.line(points, fill=(*self.hex_to_rgb(colors['accent']), 80), width=4)
        
        # Composite overlay
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        # Add blur effect to some areas for depth
        mask = Image.new('L', size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([size[0]//4, size[1]//4, size[0]*3//4, size[1]*3//4], fill=255)
        
        blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
        img = Image.composite(img, blurred, mask)
        
        # Add text with 3D effect
        draw = ImageDraw.Draw(img)
        
        try:
            font_company = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
            font_topic = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 65)
            font_industry = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 45)
        except:
            font_company = ImageFont.load_default()
            font_topic = ImageFont.load_default()
            font_industry = ImageFont.load_default()
        
        text_color = self.hex_to_rgb(colors['text'])
        shadow_color = self.hex_to_rgb(colors['shadow'])
        
        # Company name with 3D depth effect
        company_text = company.upper()
        bbox = draw.textbbox((0, 0), company_text, font=font_company)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = size[1] // 4
        
        # Create 3D depth
        depth_layers = 8
        for d in range(depth_layers, 0, -1):
            shadow_alpha = int(255 * (d / depth_layers) * 0.3)
            shadow_rgb = (*shadow_color, shadow_alpha)
            
            # Create temporary image for this layer
            temp = Image.new('RGBA', size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp)
            temp_draw.text((x+d, y+d), company_text, font=font_company, fill=shadow_rgb)
            img_rgba = img.convert('RGBA')
            img = Image.alpha_composite(img_rgba, temp).convert('RGB')
            draw = ImageDraw.Draw(img)
        
        # Main text
        draw.text((x, y), company_text, font=font_company, fill=text_color)
        
        # Glowing line effect
        line_y = y + (bbox[3] - bbox[1]) + 40
        line_width = text_width // 2
        line_x = (size[0] - line_width) // 2
        
        # Glow effect for line
        glow_img = Image.new('RGBA', size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_img)
        glow_color = self.hex_to_rgb(colors['highlight'])
        
        for thickness in range(20, 0, -2):
            alpha = int(255 * (thickness / 20) * 0.3)
            glow_draw.rectangle(
                [line_x - thickness//2, line_y - thickness//2, 
                 line_x + line_width + thickness//2, line_y + 8 + thickness//2],
                fill=(*glow_color, alpha)
            )
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, glow_img).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([line_x, line_y, line_x + line_width, line_y + 8], 
                      fill=glow_color)
        
        # Topic text with shadow
        topic_wrapped = self.wrap_text(topic, 25)
        lines = topic_wrapped.split('\n')
        y_offset = size[1] // 2 + 20
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_topic)
            text_width = bbox[2] - bbox[0]
            x = (size[0] - text_width) // 2
            
            # Soft shadow
            for offset in range(4, 0, -1):
                alpha = int(255 * (offset / 4) * 0.5)
                shadow_temp = Image.new('RGBA', size, (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow_temp)
                shadow_draw.text((x+offset, y_offset+offset), line, 
                               font=font_topic, fill=(*shadow_color, alpha))
                img_rgba = img.convert('RGBA')
                img = Image.alpha_composite(img_rgba, shadow_temp).convert('RGB')
                draw = ImageDraw.Draw(img)
            
            draw.text((x, y_offset), line, font=font_topic, fill=text_color)
            y_offset += (bbox[3] - bbox[1]) + 15
        
        # Industry badge with 3D effect
        badge_text = f"  {industry.upper()}  "
        bbox = draw.textbbox((0, 0), badge_text, font=font_industry)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        badge_x = (size[0] - text_width) // 2 - 30
        badge_y = size[1] * 3 // 4
        badge_width = text_width + 60
        badge_height = text_height + 40
        
        # 3D badge effect
        badge_color = self.hex_to_rgb(colors['accent'])
        for depth in range(10, 0, -1):
            shade_factor = 0.7 + (depth / 10) * 0.3
            shaded_color = tuple(int(c * shade_factor) for c in badge_color)
            draw.rounded_rectangle(
                [badge_x + depth, badge_y + depth, 
                 badge_x + badge_width + depth, badge_y + badge_height + depth],
                radius=25,
                fill=shaded_color
            )
        
        # Main badge
        draw.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_width, badge_y + badge_height],
            radius=25,
            fill=badge_color
        )
        
        # Badge border glow
        glow_badge = Image.new('RGBA', size, (0, 0, 0, 0))
        glow_badge_draw = ImageDraw.Draw(glow_badge)
        for thickness in range(15, 0, -1):
            alpha = int(255 * (thickness / 15) * 0.2)
            glow_badge_draw.rounded_rectangle(
                [badge_x - thickness, badge_y - thickness,
                 badge_x + badge_width + thickness, badge_y + badge_height + thickness],
                radius=25 + thickness,
                outline=(*self.hex_to_rgb(colors['highlight']), alpha),
                width=2
            )
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, glow_badge).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Badge text
        text_x = (size[0] - text_width) // 2
        text_y = badge_y + 20
        draw.text((text_x, text_y), badge_text, font=font_industry, fill=text_color)
        
        # Add decorative corner elements
        corner_size = 80
        corner_color = self.hex_to_rgb(colors['highlight'])
        
        # Top left corner
        draw.line([(20, corner_size), (20, 20), (corner_size, 20)], 
                 fill=corner_color, width=6)
        
        # Top right corner
        draw.line([(size[0]-20, corner_size), (size[0]-20, 20), (size[0]-corner_size, 20)], 
                 fill=corner_color, width=6)
        
        # Bottom left corner
        draw.line([(20, size[1]-corner_size), (20, size[1]-20), (corner_size, size[1]-20)], 
                 fill=corner_color, width=6)
        
        # Bottom right corner
        draw.line([(size[0]-20, size[1]-corner_size), (size[0]-20, size[1]-20), 
                  (size[0]-corner_size, size[1]-20)], 
                 fill=corner_color, width=6)
        
        return img
    
    def create_minimal_glass_design(self, size, company, industry, topic, colors):
        """Create glassmorphism minimal design"""
        
        # Gradient background
        gradient_start = self.hex_to_rgb(colors['gradient_start'])
        gradient_end = self.hex_to_rgb(colors['gradient_end'])
        img = self.create_radial_gradient(size, gradient_start, gradient_end)
        
        # Add blur overlay for depth
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Glass panels
        panel_color = self.hex_to_rgb(colors['primary'])
        
        # Large central glass panel
        panel_x = size[0] // 6
        panel_y = size[1] // 4
        panel_width = size[0] * 2 // 3
        panel_height = size[1] // 2
        
        # Panel shadow
        for blur_layer in range(20, 0, -2):
            alpha = int(30 * (blur_layer / 20))
            overlay_draw.rounded_rectangle(
                [panel_x + blur_layer, panel_y + blur_layer,
                 panel_x + panel_width + blur_layer, panel_y + panel_height + blur_layer],
                radius=30,
                fill=(0, 0, 0, alpha)
            )
        
        # Glass effect panel
        overlay_draw.rounded_rectangle(
            [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
            radius=30,
            fill=(*panel_color, 120)
        )
        
        # Border glow
        border_color = self.hex_to_rgb(colors['highlight'])
        overlay_draw.rounded_rectangle(
            [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
            radius=30,
            outline=(*border_color, 180),
            width=3
        )
        
        # Smaller accent panels
        accent_panels = [
            (size[0]//10, size[1]//10, 150, 150),
            (size[0]*8//10, size[1]*7//10, 180, 180)
        ]
        
        for px, py, pw, ph in accent_panels:
            overlay_draw.rounded_rectangle(
                [px, py, px + pw, py + ph],
                radius=20,
                fill=(*self.hex_to_rgb(colors['accent']), 100)
            )
            overlay_draw.rounded_rectangle(
                [px, py, px + pw, py + ph],
                radius=20,
                outline=(*border_color, 150),
                width=2
            )
        
        # Composite
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay).convert('RGB')
        
        # Apply slight blur for glass effect
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        draw = ImageDraw.Draw(img)
        
        try:
            font_company = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
            font_topic = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 55)
            font_industry = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            font_company = ImageFont.load_default()
            font_topic = ImageFont.load_default()
            font_industry = ImageFont.load_default()
        
        text_color = self.hex_to_rgb(colors['text'])
        
        # Company name centered in panel
        company_text = company.upper()
        bbox = draw.textbbox((0, 0), company_text, font=font_company)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = panel_y + 40
        
        # Text glow
        for offset in range(8, 0, -1):
            alpha = int(255 * (offset / 8) * 0.4)
            temp = Image.new('RGBA', size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp)
            glow_color = self.hex_to_rgb(colors['highlight'])
            temp_draw.text((x, y), company_text, font=font_company, 
                          fill=(*glow_color, alpha))
            temp = temp.filter(ImageFilter.GaussianBlur(radius=offset))
            img_rgba = img.convert('RGBA')
            img = Image.alpha_composite(img_rgba, temp).convert('RGB')
            draw = ImageDraw.Draw(img)
        
        draw.text((x, y), company_text, font=font_company, fill=text_color)
        
        # Topic
        topic_wrapped = self.wrap_text(topic, 30)
        bbox = draw.textbbox((0, 0), topic_wrapped, font=font_topic)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = panel_y + panel_height // 2
        draw.text((x, y), topic_wrapped, font=font_topic, fill=text_color)
        
        # Industry tag
        industry_text = industry
        bbox = draw.textbbox((0, 0), industry_text, font=font_industry)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = panel_y + panel_height - 80
        draw.text((x, y), industry_text, font=font_industry, 
                 fill=self.hex_to_rgb(colors['accent']))
        
        return img
    
    def create_bold_3d_design(self, size, company, industry, topic, colors):
        """Create bold design with strong 3D elements"""
        
        # Dynamic gradient background
        img = self.create_gradient(size, 
                                   self.hex_to_rgb(colors['primary']),
                                   self.hex_to_rgb(colors['secondary']),
                                   'vertical')
        
        # Add dynamic shapes overlay
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Large 3D spheres
        sphere_color = self.hex_to_rgb(colors['accent'])
        large_sphere = self.create_3d_sphere(400, sphere_color, (0, 0), 180)
        overlay.paste(large_sphere, (size[0] - 250, -100), large_sphere)
        
        medium_sphere = self.create_3d_sphere(300, self.hex_to_rgb(colors['highlight']), (0, 0), 150)
        overlay.paste(medium_sphere, (-100, size[1] - 200), medium_sphere)
        
        # 3D cubes scattered
        cube_positions = [
            (size[0]//4, size[1]//5, 120),
            (size[0]*3//4, size[1]*2//3, 100),
            (size[0]//2, size[1]*4//5, 90)
        ]
        
        for cx, cy, csize in cube_positions:
            cube = self.create_3d_cube(csize, self.hex_to_rgb(colors['highlight']))
            overlay.paste(cube, (cx - csize//2, cy - csize//2), cube)
        
        # Abstract geometric patterns
        accent_color = self.hex_to_rgb(colors['accent'])
        
        # Diagonal lines pattern
        for i in range(0, size[0] + size[1], 60):
            points = [(i, 0), (i - size[1], size[1])]
            overlay_draw.line(points, fill=(*accent_color, 40), width=3)
        
        # Circles pattern
        for i in range(5):
            x = random.randint(0, size[0])
            y = random.randint(0, size[1])
            radius = random.randint(80, 200)
            
            # Multi-layer circles for depth
            for r in range(radius, radius-30, -10):
                alpha = int(100 * (r / radius))
                overlay_draw.ellipse([x-r, y-r, x+r, y+r], 
                                    outline=(*self.hex_to_rgb(colors['highlight']), alpha),
                                    width=4)
        
        # Composite
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay).convert('RGB')
        
        draw = ImageDraw.Draw(img)
        
        try:
            font_company = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
            font_topic = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 65)
            font_industry = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        except:
            font_company = ImageFont.load_default()
            font_topic = ImageFont.load_default()
            font_industry = ImageFont.load_default()
        
        text_color = self.hex_to_rgb(colors['text'])
        shadow_color = self.hex_to_rgb(colors['shadow'])
        
        # Company name with extreme 3D effect
        company_text = company.upper()
        bbox = draw.textbbox((0, 0), company_text, font=font_company)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = size[1] // 4
        
        # Deep 3D shadow
        for d in range(15, 0, -1):
            shadow_x = x + d * 2
            shadow_y = y + d * 2
            brightness = 1 - (d / 15) * 0.7
            shadow_rgb = tuple(int(c * brightness) for c in shadow_color)
            draw.text((shadow_x, shadow_y), company_text, 
                     font=font_company, fill=shadow_rgb)
        
        # Outline effect
        for offset_x in range(-3, 4):
            for offset_y in range(-3, 4):
                if offset_x != 0 or offset_y != 0:
                    draw.text((x + offset_x, y + offset_y), company_text,
                             font=font_company, fill=(0, 0, 0))
        
        # Main text
        draw.text((x, y), company_text, font=font_company, fill=text_color)
        
        # Glowing accent line
        line_y = y + (bbox[3] - bbox[1]) + 50
        line_start = size[0] // 4
        line_end = size[0] * 3 // 4
        
        glow_color = self.hex_to_rgb(colors['highlight'])
        for thickness in range(30, 0, -2):
            alpha = int(255 * (thickness / 30) * 0.4)
            temp = Image.new('RGBA', size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp)
            temp_draw.line([(line_start, line_y), (line_end, line_y)],
                          fill=(*glow_color, alpha), width=thickness)
            img_rgba = img.convert('RGBA')
            img = Image.alpha_composite(img_rgba, temp).convert('RGB')
            draw = ImageDraw.Draw(img)
        
        draw.line([(line_start, line_y), (line_end, line_y)],
                 fill=glow_color, width=10)
        
        # Topic with shadow
        topic_wrapped = self.wrap_text(topic, 22)
        bbox = draw.textbbox((0, 0), topic_wrapped, font=font_topic)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = size[1] // 2
        
        # Shadow
        for offset in range(8, 0, -1):
            draw.text((x + offset, y + offset), topic_wrapped,
                     font=font_topic, fill=shadow_color)
        
        draw.text((x, y), topic_wrapped, font=font_topic, fill=text_color)
        
        # Industry badge with extreme 3D
        badge_text = f"  {industry.upper()}  "
        bbox = draw.textbbox((0, 0), badge_text, font=font_industry)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        badge_x = (size[0] - text_width) // 2 - 40
        badge_y = size[1] * 3 // 4
        badge_width = text_width + 80
        badge_height = text_height + 50
        
        # 3D badge layers
        badge_color = self.hex_to_rgb(colors['accent'])
        for depth in range(20, 0, -1):
            offset_x = badge_x + depth * 2
            offset_y = badge_y + depth * 2
            shade = 0.5 + (depth / 20) * 0.5
            shaded_color = tuple(int(c * shade) for c in badge_color)
            draw.rounded_rectangle(
                [offset_x, offset_y, offset_x + badge_width, offset_y + badge_height],
                radius=30,
                fill=shaded_color
            )
        
        # Main badge
        draw.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_width, badge_y + badge_height],
            radius=30,
            fill=badge_color
        )
        
        # Badge highlight
        highlight_color = self.hex_to_rgb(colors['highlight'])
        draw.rounded_rectangle(
            [badge_x + 5, badge_y + 5, badge_x + badge_width - 5, badge_y + badge_height - 5],
            radius=25,
            outline=highlight_color,
            width=4
        )
        
        # Badge text
        text_x = (size[0] - text_width) // 2
        text_y = badge_y + 25
        draw.text((text_x, text_y), badge_text, font=font_industry, fill=text_color)
        
        return img
    
    def create_realistic_poster(self, company, industry, topic, platform, style='modern'):
        """Generate realistic, professional poster with 3D elements"""
        
        size = self.platform_sizes.get(platform, (1080, 1080))
        colors = self.color_schemes.get(industry, self.color_schemes['Technology'])
        
        if style == 'modern_3d':
            return self.create_modern_3d_design(size, company, industry, topic, colors)
        elif style == 'glass':
            return self.create_minimal_glass_design(size, company, industry, topic, colors)
        else:  # bold_3d
            return self.create_bold_3d_design(size, company, industry, topic, colors)
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def wrap_text(self, text, max_length):
        """Wrap text to fit within max length"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_length:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def generate_complete_post(self, company, industry, topic, tone, platform, style):
        """Generate complete social media post with ultra-realistic design"""
        
        if not company or not industry or not topic:
            return None, "⚠️ Please fill in all required fields to generate your poster!", ""
        
        # Generate image with selected style
        image = self.create_realistic_poster(company, industry, topic, platform, style)
        
        # Generate comprehensive description
        description = self.generate_description(company, industry, topic, tone)
        
        # Generate strategic hashtags
        hashtags = self.generate_hashtags(company, industry, topic, platform)
        
        return image, description, hashtags

# Create generator instance
generator = SocialMediaPosterGenerator()

# Create Gradio interface with enhanced UI
with gr.Blocks(title="AI Social Media Poster Generator - Ultra Realistic", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎨 AI Social Media Poster Generator - **Ultra Realistic Edition**
    ### Create stunning, professional social media posters with advanced 3D elements, visual effects, and AI-generated content!
    
    ✨ **Features**: 3D Spheres & Cubes • Glassmorphism • Gradient Overlays • Shadow Effects • Glowing Elements • Professional Typography
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Company Information")
            company_input = gr.Textbox(
                label="Company Name",
                placeholder="Enter your company name...",
                value="TechFlow Solutions",
                info="Your brand name that will be prominently displayed"
            )
            industry_input = gr.Dropdown(
                label="Industry",
                choices=[
                    'Technology', 'Healthcare', 'Finance', 'Retail', 
                    'Food & Beverage', 'Education', 'Real Estate', 
                    'Fashion', 'Fitness', 'Travel'
                ],
                value='Technology',
                info="Select your industry for optimized color schemes"
            )
            
            gr.Markdown("### 🎯 Post Details")
            topic_input = gr.Textbox(
                label="Post Topic / Message",
                placeholder="What's your post about?",
                value="Artificial Intelligence Innovation",
                info="The main message or topic of your post"
            )
            tone_input = gr.Dropdown(
                label="Content Tone",
                choices=['Professional', 'Casual & Friendly', 'Inspiring', 'Humorous', 'Urgent'],
                value='Professional',
                info="Choose the tone for your caption and hashtags"
            )
            platform_input = gr.Dropdown(
                label="Social Media Platform",
                choices=['Instagram', 'Instagram Story', 'Facebook', 'Twitter/X', 'LinkedIn'],
                value='Instagram',
                info="Platform-optimized dimensions"
            )
            
            gr.Markdown("### 🎨 Visual Design Style")
            style_input = gr.Radio(
                label="Design Style",
                choices=[
                    ('Modern 3D - Geometric shapes with depth', 'modern_3d'),
                    ('Glass - Glassmorphism with blur effects', 'glass'),
                    ('Bold 3D - Vibrant with strong visual impact', 'bold_3d')
                ],
                value='modern_3d',
                info="Choose your preferred visual aesthetic"
            )
            
            generate_btn = gr.Button("✨ Generate Professional Poster", variant="primary", size="lg")
            
            gr.Markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-top: 20px;'>
            <p style='color: white; margin: 0;'><strong>💡 Pro Tip:</strong> Experiment with different design styles to find what resonates best with your audience!</p>
            </div>
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Your Generated Poster")
            output_image = gr.Image(
                label="Professional Social Media Poster",
                type="pil",
                show_label=True,
                show_download_button=True
            )
            
            gr.Markdown("### 📄 AI-Generated Caption")
            output_description = gr.Textbox(
                label="Complete Description with Emojis",
                lines=8,
                show_copy_button=True,
                info="Copy this caption directly to your social media post"
            )
            
            output_hashtags = gr.Textbox(
                label="Strategic Hashtags",
                lines=3,
                show_copy_button=True,
                info="Platform-optimized hashtags for maximum reach"
            )
            
            gr.Markdown("""
            <div style='background: #f0f9ff; padding: 15px; border-left: 4px solid #0ea5e9; border-radius: 5px; margin-top: 15px;'>
            <p style='margin: 0; color: #0369a1;'><strong>📱 Quick Actions:</strong></p>
            <ul style='margin: 10px 0 0 0; color: #0369a1;'>
            <li>Click download button to save your poster</li>
            <li>Use copy buttons for easy caption & hashtag copying</li>
            <li>Generate multiple versions to A/B test</li>
            </ul>
            </div>
            """)
    
    gr.Markdown("""
    ---
    ## 🎯 Design Style Guide
    
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;'>
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
            <h3 style='margin-top: 0;'>🌟 Modern 3D</h3>
            <p>• Layered geometric shapes<br>• 3D spheres and cubes<br>• Flowing curves and patterns<br>• Professional gradients<br>• Depth shadows and glows</p>
        </div>
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white;'>
            <h3 style='margin-top: 0;'>💎 Glassmorphism</h3>
            <p>• Frosted glass panels<br>• Transparent overlays<br>• Subtle blur effects<br>• Clean minimal design<br>• Border glows</p>
        </div>
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; color: white;'>
            <h3 style='margin-top: 0;'>🔥 Bold 3D</h3>
            <p>• Extreme 3D depth effects<br>• Large sphere elements<br>• Vibrant color schemes<br>• Strong visual impact<br>• Dynamic patterns</p>
        </div>
    </div>
    
    ## 📊 Platform Specifications
    
    | Platform | Dimensions | Recommended Style | Best For |
    |----------|-----------|-------------------|----------|
    | Instagram | 1080x1080 | All styles | Feed posts, engagement |
    | Instagram Story | 1080x1920 | Modern 3D, Glass | Stories, vertical content |
    | Facebook | 1200x630 | Bold 3D | Link shares, wide posts |
    | Twitter/X | 1200x675 | Modern 3D | Tweets, announcements |
    | LinkedIn | 1200x627 | Glass, Modern 3D | Professional content |
    
    ## 🚀 Content Tone Guide
    
    - **Professional**: Corporate, business-focused, expertise-driven content
    - **Casual & Friendly**: Relatable, conversational, community-building
    - **Inspiring**: Motivational, aspirational, emotional connection
    - **Humorous**: Fun, entertaining, memorable engagement
    - **Urgent**: Time-sensitive, action-driven, compelling CTAs
    """)
    
    # Connect generate button
    generate_btn.click(
        fn=generator.generate_complete_post,
        inputs=[company_input, industry_input, topic_input, tone_input, platform_input, style_input],
        outputs=[output_image, output_description, output_hashtags]
    )
    
    # Load example on startup
    app.load(
        fn=generator.generate_complete_post,
        inputs=[company_input, industry_input, topic_input, tone_input, platform_input, style_input],
        outputs=[output_image, output_description, output_hashtags]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True, server_port=9000)