import subprocess
import sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])

try:
    import torch
    import torchvision
except ImportError:
    _pip('torch', 'torchvision',
         '--index-url', 'https://download.pytorch.org/whl/cu124')  # CUDA 12.4, driver >= 525

try:
    import kagglehub
except ImportError:
    _pip('kagglehub==0.3.10')

for pkg, imp in [('opencv-python', 'cv2'), ('seaborn', 'seaborn'),
                 ('scikit-learn', 'sklearn'), ('pillow', 'PIL')]:
    try:
        __import__(imp)
    except ImportError:
        _pip(pkg)


if __name__ == '__main__':
    import time
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, roc_curve, auc)
    from sklearn.preprocessing import label_binarize
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torchvision.transforms as T
    import kagglehub

    OUT_DIR    = Path('C:/Users/lfeli/Documents/aula16_flores')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    IMG_SIZE   = 224
    BATCH_SIZE = 32
    EPOCHS     = 30
    CLASSES    = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    SEED       = 42
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {DEVICE}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print('Verificando/baixando dataset...')
    dataset_path = kagglehub.dataset_download('rahmasleam/flowers-dataset')
    DATA_DIR     = Path(dataset_path) / 'flower_photos'

    all_paths, all_labels = [], []
    for i, cls in enumerate(CLASSES):
        for p in (DATA_DIR / cls).glob('*.jpg'):
            all_paths.append(str(p))
            all_labels.append(i)

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)
    print(f'Total de imagens: {len(all_paths)}')

    # divisão estratificada 70 / 15 / 15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f'Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}')

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    train_tf = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    val_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    class FlowerDataset(Dataset):
        def __init__(self, paths, labels, transform=None):
            self.paths, self.labels, self.transform = paths, labels, transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, int(self.labels[idx])

    train_loader = DataLoader(FlowerDataset(X_train, y_train, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(FlowerDataset(X_val,   y_val,   val_tf),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(FlowerDataset(X_test,  y_test,  val_tf),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, len(CLASSES)))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    def train_epoch(model, loader, optimizer):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (out.argmax(1) == labels).sum().item()
            n          += len(imgs)
        return total_loss / n, correct / n

    @torch.no_grad()
    def eval_epoch(model, loader):
        model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * len(imgs)
            correct    += (out.argmax(1) == labels).sum().item()
            n          += len(imgs)
        return total_loss / n, correct / n

    def run_phase(model, train_loader, val_loader, optimizer, n_epochs, hist, label):
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        best_loss  = float('inf')
        patience   = 0
        best_state = None
        for epoch in range(n_epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer)
            vl_loss, vl_acc = eval_epoch(model, val_loader)
            scheduler.step(vl_loss)
            hist['loss'].append(tr_loss)
            hist['accuracy'].append(tr_acc)
            hist['val_loss'].append(vl_loss)
            hist['val_accuracy'].append(vl_acc)
            print(f'[{label}] Época {epoch+1:02d}/{n_epochs} — '
                  f'loss: {tr_loss:.4f} acc: {tr_acc:.4f} | '
                  f'val_loss: {vl_loss:.4f} val_acc: {vl_acc:.4f}')
            if vl_loss < best_loss:
                best_loss  = vl_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
                if patience >= 5:
                    print(f'Early stopping ({label})')
                    break
        model.load_state_dict(best_state)

    hist = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    t0   = time.time()

    print('\n── Fase 1: extração de características ──')
    opt1 = optim.Adam(model.classifier.parameters(), lr=1e-3)
    run_phase(model, train_loader, val_loader, opt1, EPOCHS, hist, 'F1')

    print('\n── Fase 2: fine-tuning ──')
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    opt2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    run_phase(model, train_loader, val_loader, opt2, 10, hist, 'F2')

    elapsed = (time.time() - t0) / 60
    print(f'\nTempo de treinamento e validação: {elapsed:.2f} minutos')

    @torch.no_grad()
    def predict(model, loader):
        model.eval()
        probs_all, labels_all = [], []
        for imgs, labels in loader:
            probs = torch.softmax(model(imgs.to(DEVICE)), dim=1)
            probs_all.append(probs.cpu().numpy())
            labels_all.append(np.array(labels))
        return np.concatenate(probs_all), np.concatenate(labels_all)

    y_prob, y_test_arr = predict(model, test_loader)
    y_pred = np.argmax(y_prob, axis=1)

    acc  = accuracy_score(y_test_arr, y_pred)
    prec = precision_score(y_test_arr, y_pred, average='weighted')
    rec  = recall_score(y_test_arr, y_pred, average='weighted')
    f1   = f1_score(y_test_arr, y_pred, average='weighted')

    print('\n══ Métricas de Classificação ══════════════════════════════════')
    print(f'Acurácia:  {acc:.4f}')
    print(f'Precisão:  {prec:.4f}')
    print(f'Revocação: {rec:.4f}')
    print(f'F1-Score:  {f1:.4f}')
    print(f'Tempo:     {elapsed:.2f} min')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(hist['accuracy'],     label='Treino')
    ax1.plot(hist['val_accuracy'], label='Validação')
    ax1.set_title('Acurácia'); ax1.set_xlabel('Época'); ax1.set_ylabel('Acurácia')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(hist['loss'],     label='Treino')
    ax2.plot(hist['val_loss'], label='Validação')
    ax2.set_title('Loss'); ax2.set_xlabel('Época'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'curvas_treinamento.png', dpi=150, bbox_inches='tight')
    plt.show()

    cm     = confusion_matrix(y_test_arr, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title('Matriz de Confusão (valores absolutos)')
    axes[0].set_xlabel('Predito'); axes[0].set_ylabel('Verdadeiro')
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title('Matriz de Confusão (%)')
    axes[1].set_xlabel('Predito'); axes[1].set_ylabel('Verdadeiro')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'matriz_confusao.png', dpi=150, bbox_inches='tight')
    plt.show()

    PAIRS = [
        (0, 1, 'daisy vs dandelion'),
        (1, 2, 'dandelion vs roses'),
        (2, 3, 'roses vs sunflowers'),
        (3, 4, 'sunflowers vs tulips'),
        (4, 0, 'tulips vs daisy'),
    ]
    y_bin = label_binarize(y_test_arr, classes=list(range(len(CLASSES))))
    plt.figure(figsize=(9, 7))
    for i, j, label in PAIRS:
        mask = np.isin(y_test_arr, [i, j])
        fpr, tpr, _ = roc_curve(y_bin[mask, i], y_prob[mask, i])
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC={auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('Taxa de Falso Positivo (FPR)'); plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
    plt.title('Curvas ROC – Par a Par entre Classes')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / 'roc_auc.png', dpi=150, bbox_inches='tight')
    plt.show()

    hits   = np.diag(cm)
    total  = cm.sum(axis=1)
    misses = total - hits
    x = np.arange(len(CLASSES)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w / 2, hits,   w, label='Acertos', color='steelblue')
    b2 = ax.bar(x + w / 2, misses, w, label='Erros',   color='tomato')
    for bar, v, t in zip(b1, hits, total):
        ax.text(bar.get_x() + w / 2, bar.get_height() + 0.3,
                f'{v}\n({v / t * 100:.1f}%)', ha='center', va='bottom', fontsize=9)
    for bar, v, t in zip(b2, misses, total):
        ax.text(bar.get_x() + w / 2, bar.get_height() + 0.3,
                f'{v}\n({v / t * 100:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylabel('Quantidade de Imagens')
    ax.set_title('Acertos e Erros por Classe (conjunto de teste)')
    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / 'acertos_erros.png', dpi=150, bbox_inches='tight')
    plt.show()

    class GradCAM:
        def __init__(self, model):
            self.activation = self.gradient = None
            target = model.features[-1]  # última camada conv do EfficientNetB0
            self._fh = target.register_forward_hook(
                lambda m, i, o: setattr(self, 'activation', o.detach()))
            self._bh = target.register_full_backward_hook(
                lambda m, gi, go: setattr(self, 'gradient', go[0].detach()))
            self.model = model

        def __call__(self, img_tensor):
            self.model.eval()
            out = self.model(img_tensor.unsqueeze(0).to(DEVICE))
            self.model.zero_grad()
            out[0, out.argmax(1).item()].backward()
            weights = self.gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activation).sum(dim=1))[0].cpu().numpy()
            cam = cam / (cam.max() + 1e-8)
            return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        def remove(self):
            self._fh.remove(); self._bh.remove()

    gradcam = GradCAM(model)

    class_imgs = {i: [] for i in range(len(CLASSES))}
    for path, lbl in zip(X_test, y_test_arr):
        lbl = int(lbl)
        if len(class_imgs[lbl]) < 3:
            class_imgs[lbl].append(path)
        if all(len(v) == 3 for v in class_imgs.values()):
            break

    fig, axes = plt.subplots(5, 3, figsize=(11, 19))
    for cls_i, cls_name in enumerate(CLASSES):
        for img_i, img_path in enumerate(class_imgs[cls_i]):
            img_t   = val_tf(Image.open(img_path).convert('RGB'))
            cam     = gradcam(img_t)
            img_vis = img_t.clone()
            for c in range(3):
                img_vis[c] = img_t[c] * (1.0 / STD[c]) + (-MEAN[c] / STD[c])
            img_vis = img_vis.permute(1, 2, 0).clamp(0, 1).numpy()
            heatmap = cv2.cvtColor(
                cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            axes[cls_i][img_i].imshow(np.clip(0.6 * img_vis + 0.4 * heatmap, 0, 1))
            axes[cls_i][img_i].set_title(f'{cls_name} [{img_i + 1}]', fontsize=10)
            axes[cls_i][img_i].axis('off')

    gradcam.remove()
    plt.suptitle('Grad-CAM – 3 imagens por classe (conjunto de teste)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'gradcam.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'\nGráficos salvos em: {OUT_DIR}')
