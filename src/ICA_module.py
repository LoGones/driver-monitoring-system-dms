import numpy as np

def whiten_rows(X):
    """Sbianca i dati per rimuovere la correlazione tra i canali RGB."""
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    cov = np.cov(X_centered)
    d, E = np.linalg.eigh(cov)

    # Ordina gli autovalori in modo decrescente
    idx = np.argsort(d)[::-1]
    d = d[idx]
    E = E[:, idx]

    V = np.diag(1.0 / np.sqrt(d + 1e-8)) @ E.T  # 1e-8 per evitare divisioni per zero
    Z = V @ X_centered
    return Z, V


def fastica_deflation(Z, n_components=3, max_iter=1000, tol=1e-7):
    """Applica FastICA per separare le sorgenti indipendenti."""
    m, N = Z.shape
    W = np.zeros((n_components, m))

    for p in range(n_components):
        w = np.random.randn(m)
        w /= np.linalg.norm(w)

        for i in range(max_iter):
            w_old = w.copy()

            y = w @ Z
            g = np.tanh(y)
            gp = 1 - g ** 2

            # Aggiornamento FastICA
            w = (Z @ g.T) / N - np.mean(gp) * w_old

            # Ortogonalizzazione
            if p > 0:
                w -= W[:p].T @ (W[:p] @ w)

            w /= np.linalg.norm(w)

            if np.abs(np.abs(w @ w_old) - 1) < tol:
                break

        W[p, :] = w

    return W @ Z


def extract_hr_from_rgb(r_channel, g_channel, b_channel, fs):
    """
    Stima la frequenza cardiaca partendo dai segnali RGB di un video.

    Parametri:
    r_channel, g_channel, b_channel: array numpy 1D (le medie del colore nel tempo)
    fs: float, i frame per secondo (FPS) del video

    Ritorna:
    float: i Battiti Per Minuto (BPM) stimati
    """
    # 1. Creazione della matrice delle osservazioni
    X = np.vstack([r_channel, g_channel, b_channel])

    # 2. Sbiancamento
    Z, _ = whiten_rows(X)

    # 3. Estrazione delle Componenti Indipendenti (ICA)
    S_est = fastica_deflation(Z, n_components=3)

    # 4. Selezione Automatica della componente cardiaca
    best_hr = 0
    max_power = -1

    # Definiamo il range fisiologico umano: da 45 a 180 BPM
    min_hz = 45 / 60.0
    max_hz = 180 / 60.0

    # Analizziamo tutte e 3 le componenti estratte per trovare quella col battito
    for i in range(3):
        signal = S_est[i, :]

        # Trasformata di Fourier
        f_signal = np.fft.fft(signal)
        psd = np.abs(f_signal) ** 2  # Spettro di potenza
        freqs = np.fft.fftfreq(len(signal), 1 / fs)

        # Consideriamo solo le frequenze positive nel range fisiologico
        valid_idx = np.where((freqs >= min_hz) & (freqs <= max_hz))[0]

        if len(valid_idx) == 0:
            continue

        valid_freqs = freqs[valid_idx]
        valid_psd = psd[valid_idx]

        # Troviamo il picco massimo in questo range
        peak_idx = np.argmax(valid_psd)
        peak_power = valid_psd[peak_idx]

        # Se questo picco è il più forte visto finora tra le componenti, lo salviamo
        if peak_power > max_power:
            max_power = peak_power
            best_hr = valid_freqs[peak_idx] * 60.0 # BPM conversion

    return best_hr