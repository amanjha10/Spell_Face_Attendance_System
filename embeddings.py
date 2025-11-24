import os
import pickle
import numpy as np
from PIL import Image
import pillow_heif
import cv2
from insightface.app import FaceAnalysis


class EmbeddingsGenerator:
    def __init__(self):
        self.app = None
        pillow_heif.register_heif_opener()
        self._init_model()
    
    def _init_model(self):
        """Initialize InsightFace model"""
        print("🤖 Initializing InsightFace model...")
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ Model initialized")
    
    def _load_image(self, image_path):
        """Load image from file (supports HEIC, JPEG, PNG)"""
        try:
            if image_path.lower().endswith(('.heic', '.HEIC')):
                # Handle HEIC files
                pil_image = Image.open(image_path)
                if pil_image.mode in ('RGBA', 'LA'):
                    pil_image = pil_image.convert('RGB')
                rgb_array = np.array(pil_image)
                return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            else:

                return cv2.imread(image_path)
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
            return None
    
    def _extract_face_embedding(self, image_path):
        """Extract face embedding from single image"""
        image = self._load_image(image_path)
        if image is None:
            return None
        
        try:
            faces = self.app.get(image)
            if faces:
                return faces[0].normed_embedding
            return None
        except Exception as e:
            print(f"❌ Error extracting face from {image_path}: {e}")
            return None
    
    def _process_person_folder(self, person_name, folder_path):
        """Process all images in a person's folder"""
        print(f"\n👤 Processing {person_name}...")
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return []
        
        embeddings = []
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.heic', '.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No image files found in {folder_path}")
            return []
        
        print(f"📸 Found {len(image_files)} images")
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, filename)
            print(f"   Processing {i}/{len(image_files)}: {filename}")
            
            embedding = self._extract_face_embedding(image_path)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"   ✅ Face detected and processed")
            else:
                print(f"   ❌ No face detected")
        
        print(f"✅ {person_name}: {len(embeddings)}/{len(image_files)} images processed successfully")
        return embeddings
    
    def generate_all_embeddings(self):
        """Generate embeddings for all people in data folder"""
        print("🚀 Starting embeddings generation for all people...")
        print("=" * 60)
        
        data_folder = "data"
        if not os.path.exists(data_folder):
            print(f"❌ Data folder not found: {data_folder}")
            return
        
        # Create embeddings directory
        os.makedirs('embeddings', exist_ok=True)
        
        # Find all person folders
        person_folders = [f for f in os.listdir(data_folder) 
                         if os.path.isdir(os.path.join(data_folder, f))]
        
        if not person_folders:
            print(f"❌ No person folders found in {data_folder}")
            return
        
        print(f"📁 Found {len(person_folders)} person folders: {person_folders}")
        
        all_embeddings = {}
        employee_counter = 1
        
        # Process each person
        for person_name in person_folders:
            folder_path = os.path.join(data_folder, person_name)
            embeddings = self._process_person_folder(person_name, folder_path)
            
            if embeddings:
                # Generate employee ID
                emp_id = f"EMP{employee_counter:03d}"  # EMP001, EMP002, etc.
                
                # Ask for email address
                print(f"\n📧 Email setup for {person_name} ({emp_id}):")
                email = input(f"Enter email address for {person_name}: ").strip()
                while email and '@' not in email:
                    print("❌ Please enter a valid email address")
                    email = input(f"Enter email address for {person_name}: ").strip()
                
                if not email:
                    email = f"{person_name.lower()}@example.com"  # Default email
                    print(f"⚠️  Using default email: {email}")
                
                all_embeddings[person_name] = {
                    "employee_id": emp_id,
                    "email": email,
                    "embeddings": embeddings
                }
                employee_counter += 1
                print(f"💾 Saved {len(embeddings)} embeddings for {person_name} ({emp_id}) - {email}")
            else:
                print(f"⚠️  No embeddings generated for {person_name}")
        
        # Save to file
        if all_embeddings:
            embeddings_file = 'embeddings/faces.pkl'
            with open(embeddings_file, 'wb') as f:
                pickle.dump(all_embeddings, f)
            
            print("\n" + "=" * 60)
            print("✅ EMBEDDINGS GENERATION COMPLETE!")
            print(f"💾 Saved to: {embeddings_file}")
            print("\n📊 Summary:")
            for name, data in all_embeddings.items():
                print(f"   👤 {name} ({data['employee_id']}): {len(data['embeddings'])} embeddings - {data['email']}")
            print(f"\n🎯 Total people: {len(all_embeddings)}")
            print("=" * 60)
        else:
            print("❌ No embeddings were generated for any person")
    
    def update_single_person(self, person_name):
        """Update embeddings for a single person (useful for adding new photos)"""
        print(f"🔄 Updating embeddings for {person_name}...")
        
        # Load existing embeddings
        embeddings_file = 'embeddings/faces.pkl'
        all_embeddings = {}
        
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                all_embeddings = pickle.load(f)
            print(f"📂 Loaded existing embeddings for {len(all_embeddings)} people")
        
        # Process the specific person
        folder_path = os.path.join("data", person_name)
        embeddings = self._process_person_folder(person_name, folder_path)
        
        if embeddings:
            # Determine employee ID and email
            if person_name in all_embeddings:
                emp_id = all_embeddings[person_name]['employee_id']
                existing_email = all_embeddings[person_name].get('email', '')
                print(f"🔄 Updating existing person: {person_name} ({emp_id})")
                print(f"Current email: {existing_email}")
                
                email = input(f"Update email for {person_name} (press Enter to keep current): ").strip()
                if not email:
                    email = existing_email
            else:
                # New person - generate new employee ID
                existing_ids = [data['employee_id'] for data in all_embeddings.values()]
                next_num = len(existing_ids) + 1
                emp_id = f"EMP{next_num:03d}"
                print(f"🆕 Adding new person: {person_name} ({emp_id})")
                
                email = input(f"Enter email address for {person_name}: ").strip()
                while email and '@' not in email:
                    print("❌ Please enter a valid email address")
                    email = input(f"Enter email address for {person_name}: ").strip()
                
                if not email:
                    email = f"{person_name.lower()}@example.com"
                    print(f"⚠️  Using default email: {email}")
            
            all_embeddings[person_name] = {
                "employee_id": emp_id,
                "email": email,
                "embeddings": embeddings
            }
            
            # Save updated embeddings
            with open(embeddings_file, 'wb') as f:
                pickle.dump(all_embeddings, f)
            
            print(f"✅ Updated {person_name} with {len(embeddings)} embeddings")
        else:
            print(f"❌ No embeddings generated for {person_name}")


def main():
    """Main function with menu options"""
    generator = EmbeddingsGenerator()
    
    print("🎯 Face Embeddings Generator")
    print("=" * 40)
    print("1. 🔄 Generate embeddings for ALL people")
    print("2. 👤 Update embeddings for ONE person")
    print("3. 📧 Update email addresses only")
    print("4. 📊 View current embeddings info")
    print("5. 🚪 Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        generator.generate_all_embeddings()
    
    elif choice == '2':
        data_folder = "data"
        if os.path.exists(data_folder):
            person_folders = [f for f in os.listdir(data_folder) 
                            if os.path.isdir(os.path.join(data_folder, f))]
            
            if person_folders:
                print(f"\nAvailable people: {person_folders}")
                person_name = input("Enter person name: ").strip()
                
                if person_name in person_folders:
                    generator.update_single_person(person_name)
                else:
                    print(f"❌ Person '{person_name}' not found in data folder")
            else:
                print("❌ No person folders found in data/")
        else:
            print("❌ Data folder not found")
    
    elif choice == '3':
        # Update email addresses only
        embeddings_file = 'embeddings/faces.pkl'
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n📧 Update Email Addresses")
            print("=" * 40)
            
            updated = False
            for name, info in data.items():
                current_email = info.get('email', 'No email set')
                emp_id = info['employee_id']
                
                print(f"\n👤 {name} ({emp_id})")
                print(f"Current email: {current_email}")
                
                new_email = input(f"Enter new email for {name} (press Enter to keep current): ").strip()
                
                if new_email:
                    while '@' not in new_email:
                        print("❌ Please enter a valid email address")
                        new_email = input(f"Enter new email for {name}: ").strip()
                        if not new_email:  # User pressed Enter to skip
                            break
                    
                    if new_email and new_email != current_email:
                        data[name]['email'] = new_email
                        print(f"✅ Updated {name}'s email to: {new_email}")
                        updated = True
                    elif new_email:
                        print(f"ℹ️  Email unchanged for {name}")
                else:
                    print(f"ℹ️  Email unchanged for {name}")
            
            if updated:
                # Save updated data
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(data, f)
                print(f"\n✅ Email addresses updated successfully!")
                
                # Show summary
                print("\n📊 Updated Email Summary:")
                print("=" * 30)
                for name, info in data.items():
                    email = info.get('email', 'No email set')
                    print(f"👤 {name} ({info['employee_id']}): {email}")
            else:
                print("\n📧 No email addresses were updated.")
        else:
            print("❌ No embeddings file found. Run option 1 first.")
    
    elif choice == '4':
        embeddings_file = 'embeddings/faces.pkl'
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n📊 Current Embeddings Database:")
            print("=" * 40)
            for name, info in data.items():
                email = info.get('email', 'No email set')
                print(f"👤 {name} ({info['employee_id']}): {len(info['embeddings'])} embeddings - {email}")
            print(f"\n🎯 Total people: {len(data)}")
        else:
            print("❌ No embeddings file found. Run option 1 first.")
    
    elif choice == '5':
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
